# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for deterministic TensorFlow operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
from absl.testing import parameterized
from compare_gan.tpu import tpu_random
import numpy as np
from six.moves import range
import tensorflow as tf


FLAGS = flags.FLAGS


class TpuRandomTest(parameterized.TestCase, tf.test.TestCase):

  def _run_graph_op_in_estimator(self, create_op_fn, model_dir, use_tpu,
                                 training_steps=4):
    """Helper function to test an operation within a Estimator.

    Args:
      create_op_fn: Function that will be called from within the model_fn.
        The returned op will be run as part of the training step.
      model_dir: Directory for saving checkpoints.
      use_tpu: Whether to use TPU.
      training_steps: Number of trainings steps.
    """
    def input_fn(params):
      features = {"x": np.ones((8, 3), dtype=np.float32)}
      labels = np.ones((8, 1), dtype=np.float32)
      dataset = tf.data.Dataset.from_tensor_slices((features, labels))
      # Add a feature for the random offset of operations in tpu_random.py.
      dataset = tpu_random.add_random_offset_to_features(dataset)
      return dataset.repeat().batch(params["batch_size"], drop_remainder=True)

    def model_fn(features, labels, mode, params):
      # Set the random offset tensor for operations in tpu_random.py.
      tpu_random.set_random_offset_from_features(features)
      test_op = create_op_fn()
      predictions = tf.layers.dense(features["x"], 1)
      loss = tf.losses.mean_squared_error(labels, predictions)
      optimizer = tf.train.GradientDescentOptimizer(0.01)
      if params["use_tpu"]:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
      with tf.control_dependencies([test_op]):
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_or_create_global_step())
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op)

    if tf.gfile.Exists(model_dir):
      tf.gfile.DeleteRecursively(model_dir)
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=1,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    estimator = tf.contrib.tpu.TPUEstimator(
        config=run_config,
        use_tpu=use_tpu,
        model_fn=model_fn,
        train_batch_size=2)
    estimator.train(input_fn, steps=training_steps)

  @parameterized.parameters(
      {"use_tpu": False},
      {"use_tpu": True},
  )
  def testIsDeterministic(self, use_tpu):
    def create_op_fn():
      z = tf.get_variable("z", (3,), tf.float32)
      random_z = tpu_random.uniform((3,), name="random_z")
      if use_tpu:
        random_z = tf.contrib.tpu.cross_replica_sum(random_z)
      return tf.assign(z, random_z).op

    model_dir_1 = os.path.join(FLAGS.test_tmpdir, "1")
    self._run_graph_op_in_estimator(create_op_fn, model_dir_1, use_tpu=use_tpu)

    model_dir_2 = os.path.join(FLAGS.test_tmpdir, "2")
    self._run_graph_op_in_estimator(create_op_fn, model_dir_2, use_tpu=use_tpu)

    for step in range(1, 5):
      self.assertTrue(tf.gfile.Exists(
          os.path.join(model_dir_1, "model.ckpt-{}.index".format(step))))
      ckpt_1 = tf.train.load_checkpoint(
          os.path.join(model_dir_1, "model.ckpt-{}".format(step)))
      ckpt_2 = tf.train.load_checkpoint(
          os.path.join(model_dir_2, "model.ckpt-{}".format(step)))
      z_1 = ckpt_1.get_tensor("z")
      z_2 = ckpt_2.get_tensor("z")
      logging.info("step=%d, z_1=%s, z_2=%s", step, z_1, z_2)
      # Both runs are the same.
      self.assertAllClose(z_1, z_2)

  @parameterized.parameters(
      {"use_tpu": False},
      {"use_tpu": True},
  )
  def testIsDifferentAcrossSteps(self, use_tpu):
    def create_op_fn():
      z = tf.get_variable("z", (3,), tf.float32)
      random_z = tpu_random.uniform((3,), name="random_z")
      if use_tpu:
        random_z = tf.contrib.tpu.cross_replica_sum(random_z)
      return tf.assign(z, random_z).op

    model_dir = os.path.join(FLAGS.test_tmpdir, "1")
    self._run_graph_op_in_estimator(create_op_fn, model_dir, use_tpu=use_tpu)

    previous_z = None
    for step in range(1, 5):
      self.assertTrue(tf.gfile.Exists(
          os.path.join(model_dir, "model.ckpt-{}.index".format(step))))
      ckpt = tf.train.load_checkpoint(
          os.path.join(model_dir, "model.ckpt-{}".format(step)))
      z = ckpt.get_tensor("z")
      logging.info("step=%d, z=%s", step, z)
      # Different to previous run.
      if previous_z is not None:
        self.assertNotAllClose(previous_z, z)
      previous_z = z

  def testIsDifferentAcrossCores(self):
    def create_op_fn():
      z_sum = tf.get_variable("z_sum", (3,), tf.float32)
      z_first_core = tf.get_variable("z_first_core", (3,), tf.float32)
      random_z = tpu_random.uniform((3,), name="random_z")
      random_z_sum = tf.contrib.tpu.cross_replica_sum(random_z)
      return tf.group(tf.assign(z_sum, random_z_sum).op,
                      tf.assign(z_first_core, random_z))

    model_dir = os.path.join(FLAGS.test_tmpdir, "1")
    self._run_graph_op_in_estimator(create_op_fn, model_dir, use_tpu=True)

    for step in range(1, 5):
      self.assertTrue(tf.gfile.Exists(
          os.path.join(model_dir, "model.ckpt-{}.index".format(step))))
      ckpt = tf.train.load_checkpoint(
          os.path.join(model_dir, "model.ckpt-{}".format(step)))
      z_sum = ckpt.get_tensor("z_sum")
      z_first_core = ckpt.get_tensor("z_first_core")
      logging.info("step=%d, z_sum=%s, z_first_core=%s",
                   step, z_sum, z_first_core)
      # Sum is not the first core times 2.
      self.assertNotAllClose(z_sum, 2 * z_first_core)


if __name__ == "__main__":
  tf.test.main()
