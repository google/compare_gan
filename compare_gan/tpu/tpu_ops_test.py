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

"""Tests custom TensorFlow operations for TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import parameterized
from compare_gan.tpu import tpu_ops
import numpy as np
import tensorflow as tf


class TpuOpsTpuTest(parameterized.TestCase, tf.test.TestCase):

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
      logging.info("devices:\n%s", "\n".join([str(d) for d in devices]))
      self.assertAllEqual([d.name for d in devices], expected_device_names)

  def testCrossReplicaConcat(self):
    def computation(x, replica_id):
      logging.info("x: %s\nreplica_id: %s", x, replica_id[0])
      return tpu_ops.cross_replica_concat(x, replica_id[0], num_replicas=2)

    inputs = np.asarray([[3, 4], [1, 5]])
    expected_output = np.asarray([[3, 4], [1, 5], [3, 4], [1, 5]])

    with tf.Graph().as_default():
      x = tf.constant(inputs)
      replica_ids = tf.constant([0, 1], dtype=tf.int32)
      x_concat, = tf.contrib.tpu.batch_parallel(
          computation, [x, replica_ids], num_shards=2)
      self.assertAllEqual(x.shape.as_list(), [2, 2])
      self.assertAllEqual(x_concat.shape.as_list(), [4, 2])

      with self.session() as sess:
        sess.run(tf.contrib.tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        x_concat = sess.run(x_concat)
        logging.info("x_concat: %s", x_concat)
        self.assertAllClose(x_concat, expected_output)

  # Test with group size 2 (test case has 2 cores, so this global batch norm).
  @parameterized.parameters(
      {"group_size": None},  # Defaults to number of TPU cores.
      {"group_size": 0},  # Defaults to number of TPU cores.
      {"group_size": 2},
  )
  def testCrossReplicaMean(self, group_size):
    # Verify that we average across replicas by feeding 2 vectors to the system.
    # Each replica should get one vector which is then averaged across
    # all replicas and simply returned.
    # After that each replica has the same vector and since the outputs gets
    # concatenated we see the same vector twice.
    inputs = np.asarray(
        [[0.55, 0.70, -1.29, 0.502], [0.57, 0.90, 1.290, 0.202]],
        dtype=np.float32)
    expected_output = np.asarray(
        [[0.56, 0.8, 0.0, 0.352], [0.56, 0.8, 0.0, 0.352]], dtype=np.float32)

    def computation(x):
      self.assertAllEqual(x.shape.as_list(), [1, 4])
      return tpu_ops.cross_replica_mean(x, group_size=group_size)

    with tf.Graph().as_default():
      # Note: Using placeholders for feeding TPUs is discouraged but fine for
      # a simple test case.
      x = tf.placeholder(name="x", dtype=tf.float32, shape=inputs.shape)
      y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=2)
      with self.session() as sess:
        sess.run(tf.contrib.tpu.initialize_system())
        # y is actually a list with one tensor. computation would be allowed
        # to return multiple tensors (and ops).
        actual_output = sess.run(y, {x: inputs})[0]

    self.assertAllEqual(actual_output.shape, (2, 4))
    self.assertAllClose(actual_output, expected_output)

  def testCrossReplicaMeanGroupSizeOne(self, group_size=1):
    # Since the group size is 1 we only average over 1 replica.
    inputs = np.asarray(
        [[0.55, 0.70, -1.29, 0.502], [0.57, 0.90, 1.290, 0.202]],
        dtype=np.float32)
    expected_output = np.asarray(
        [[0.55, 0.7, -1.29, 0.502], [0.57, 0.9, 1.290, 0.202]],
        dtype=np.float32)

    def computation(x):
      self.assertAllEqual(x.shape.as_list(), [1, 4])
      return tpu_ops.cross_replica_mean(x, group_size=group_size)

    with tf.Graph().as_default():
      # Note: Using placeholders for feeding TPUs is discouraged but fine for
      # a simple test case.
      x = tf.placeholder(name="x", dtype=tf.float32, shape=inputs.shape)
      y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=2)
      with self.session() as sess:
        sess.run(tf.contrib.tpu.initialize_system())
        # y is actually a list with one tensor. computation would be allowed
        # to return multiple tensors (and ops).
        actual_output = sess.run(y, {x: inputs})[0]

    self.assertAllEqual(actual_output.shape, (2, 4))
    self.assertAllClose(actual_output, expected_output)


if __name__ == "__main__":
  tf.test.main()
