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

"""Tests for custom architecture operations on TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from compare_gan.architectures import arch_ops
import gin
import numpy as np
import tensorflow as tf


class ArchOpsTpuTest(tf.test.TestCase):

  def setUp(self):
    # Construct input for batch norm tests:
    # 4 images with resolution 2x1 and 3 channels.
    x1 = np.asarray([[[5, 7, 2]], [[5, 8, 8]]], dtype=np.float32)
    x2 = np.asarray([[[1, 2, 0]], [[4, 0, 4]]], dtype=np.float32)
    x3 = np.asarray([[[6, 2, 6]], [[5, 0, 5]]], dtype=np.float32)
    x4 = np.asarray([[[2, 4, 2]], [[6, 4, 1]]], dtype=np.float32)
    self._inputs = np.stack([x1, x2, x3, x4])
    self.assertAllEqual(self._inputs.shape, [4, 2, 1, 3])
    # And the expected output for applying batch norm (without additional
    # scaling/shifting).
    self._expected_outputs = np.asarray(
        [[[[0.4375205, 1.30336881, -0.58830315]],
          [[0.4375205, 1.66291881, 1.76490951]]],
         [[[-1.89592218, -0.49438119, -1.37270737]],
          [[-0.14584017, -1.21348119, 0.19610107]]],
         [[[1.02088118, -0.49438119, 0.98050523]],
          [[0.4375205, -1.21348119, 0.58830321]]],
         [[[-1.31256151, 0.22471881, -0.58830315]],
          [[1.02088118, 0.22471881, -0.98050523]]]],
        dtype=np.float32)
    self.assertAllEqual(self._expected_outputs.shape, [4, 2, 1, 3])

  def testRunsOnTpu(self):
    """Verify that the test cases runs on a TPU chip and has 2 cores."""
    expected_device_names = [
        "/job:localhost/replica:0/task:0/device:CPU:0",
        "/job:localhost/replica:0/task:0/device:TPU:0",
        "/job:localhost/replica:0/task:0/device:TPU:1",
        "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
    ]
    with self.session() as sess:
      devices = sess.list_devices()
      tf.logging.info("devices:\n%s", "\n".join([str(d) for d in devices]))
      self.assertAllEqual([d.name for d in devices], expected_device_names)

  def testBatchNormOneCore(self):
    def computation(x):
      core_bn = tf.layers.batch_normalization(x, training=True)
      contrib_bn = tf.contrib.layers.batch_norm(x, is_training=True)
      custom_bn = arch_ops.batch_norm(x, is_training=True)
      tf.logging.info("custom_bn tensor: %s", custom_bn)
      return core_bn, contrib_bn, custom_bn

    with tf.Graph().as_default():
      x = tf.constant(self._inputs)
      core_bn, contrib_bn, custom_bn = tf.contrib.tpu.batch_parallel(
          computation, [x], num_shards=1)

      with self.session() as sess:
        sess.run(tf.contrib.tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        core_bn, contrib_bn, custom_bn = sess.run(
            [core_bn, contrib_bn, custom_bn])
        logging.info("core_bn: %s", core_bn)
        logging.info("contrib_bn: %s", contrib_bn)
        logging.info("custom_bn: %s", custom_bn)
        self.assertAllClose(core_bn, self._expected_outputs)
        self.assertAllClose(contrib_bn, self._expected_outputs)
        self.assertAllClose(custom_bn, self._expected_outputs)

  def testBatchNormTwoCoresCoreAndContrib(self):
    def computation(x):
      core_bn = tf.layers.batch_normalization(x, training=True)
      contrib_bn = tf.contrib.layers.batch_norm(x, is_training=True)
      return core_bn, contrib_bn

    with tf.Graph().as_default():
      x = tf.constant(self._inputs)
      core_bn, contrib_bn = tf.contrib.tpu.batch_parallel(
          computation, [x], num_shards=2)

      with self.session() as sess:
        sess.run(tf.contrib.tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        core_bn, contrib_bn = sess.run([core_bn, contrib_bn])
        logging.info("core_bn: %s", core_bn)
        logging.info("contrib_bn: %s", contrib_bn)
        self.assertNotAllClose(core_bn, self._expected_outputs)
        self.assertNotAllClose(contrib_bn, self._expected_outputs)

  def testBatchNormTwoCoresCustom(self):
    def computation(x):
      custom_bn = arch_ops.batch_norm(x, is_training=True, name="custom_bn")
      gin.bind_parameter("cross_replica_moments.parallel", False)
      custom_bn_seq = arch_ops.batch_norm(x, is_training=True,
                                          name="custom_bn_seq")
      return custom_bn, custom_bn_seq

    with tf.Graph().as_default():
      x = tf.constant(self._inputs)
      custom_bn, custom_bn_seq = tf.contrib.tpu.batch_parallel(
          computation, [x], num_shards=2)

      with self.session() as sess:
        sess.run(tf.contrib.tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        custom_bn, custom_bn_seq = sess.run(
            [custom_bn, custom_bn_seq])
        logging.info("custom_bn: %s", custom_bn)
        logging.info("custom_bn_seq: %s", custom_bn_seq)
        self.assertAllClose(custom_bn, self._expected_outputs)
        self.assertAllClose(custom_bn_seq, self._expected_outputs)


if __name__ == "__main__":
  tf.test.main()
