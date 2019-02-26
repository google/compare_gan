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

"""Tests for custom architecture operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import arch_ops
import numpy as np
import tensorflow as tf


class ArchOpsTest(tf.test.TestCase):

  def testBatchNorm(self):
    with tf.Graph().as_default():
      # 4 images with resolution 2x1 and 3 channels.
      x1 = tf.constant([[[5, 7, 2]], [[5, 8, 8]]], dtype=tf.float32)
      x2 = tf.constant([[[1, 2, 0]], [[4, 0, 4]]], dtype=tf.float32)
      x3 = tf.constant([[[6, 2, 6]], [[5, 0, 5]]], dtype=tf.float32)
      x4 = tf.constant([[[2, 4, 2]], [[6, 4, 1]]], dtype=tf.float32)
      x = tf.stack([x1, x2, x3, x4])
      self.assertAllEqual(x.shape.as_list(), [4, 2, 1, 3])

      core_bn = tf.layers.batch_normalization(x, training=True)
      contrib_bn = tf.contrib.layers.batch_norm(x, is_training=True)
      custom_bn = arch_ops.batch_norm(x, is_training=True)
      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        core_bn, contrib_bn, custom_bn = sess.run(
            [core_bn, contrib_bn, custom_bn])
        tf.logging.info("core_bn: %s", core_bn[0])
        tf.logging.info("contrib_bn: %s", contrib_bn[0])
        tf.logging.info("custom_bn: %s", custom_bn[0])
        self.assertAllClose(core_bn, contrib_bn)
        self.assertAllClose(custom_bn, contrib_bn)
        expected_values = np.asarray(
            [[[[0.4375205, 1.30336881, -0.58830315]],
              [[0.4375205, 1.66291881, 1.76490951]]],
             [[[-1.89592218, -0.49438119, -1.37270737]],
              [[-0.14584017, -1.21348119, 0.19610107]]],
             [[[1.02088118, -0.49438119, 0.98050523]],
              [[0.4375205, -1.21348119, 0.58830321]]],
             [[[-1.31256151, 0.22471881, -0.58830315]],
              [[1.02088118, 0.22471881, -0.98050523]]]],
            dtype=np.float32)
        self.assertAllClose(custom_bn, expected_values)

  def testAccumulatedMomentsDuringTraing(self):
    with tf.Graph().as_default():
      mean_in = tf.placeholder(tf.float32, shape=[2])
      variance_in = tf.placeholder(tf.float32, shape=[2])
      mean, variance = arch_ops._accumulated_moments_for_inference(
          mean=mean_in, variance=variance_in, is_training=True)
      variables_by_name = {v.op.name: v for v in tf.global_variables()}
      tf.logging.error(variables_by_name)
      accu_mean = variables_by_name["accu/accu_mean"]
      accu_variance = variables_by_name["accu/accu_variance"]
      accu_counter = variables_by_name["accu/accu_counter"]
      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, v1 = sess.run(
            [mean, variance],
            feed_dict={mean_in: [1.0, 2.0], variance_in: [3.0, 4.0]})
        self.assertAllClose(m1, [1.0, 2.0])
        self.assertAllClose(v1, [3.0, 4.0])
        m2, v2 = sess.run(
            [mean, variance],
            feed_dict={mean_in: [5.0, 6.0], variance_in: [7.0, 8.0]})
        self.assertAllClose(m2, [5.0, 6.0])
        self.assertAllClose(v2, [7.0, 8.0])
        am, av, ac = sess.run([accu_mean, accu_variance, accu_counter])
        self.assertAllClose(am, [0.0, 0.0])
        self.assertAllClose(av, [0.0, 0.0])
        self.assertAllClose([ac], [0.0])

  def testAccumulatedMomentsDuringEal(self):
    with tf.Graph().as_default():
      mean_in = tf.placeholder(tf.float32, shape=[2])
      variance_in = tf.placeholder(tf.float32, shape=[2])
      mean, variance = arch_ops._accumulated_moments_for_inference(
          mean=mean_in, variance=variance_in, is_training=False)
      variables_by_name = {v.op.name: v for v in tf.global_variables()}
      tf.logging.error(variables_by_name)
      accu_mean = variables_by_name["accu/accu_mean"]
      accu_variance = variables_by_name["accu/accu_variance"]
      accu_counter = variables_by_name["accu/accu_counter"]
      update_accus = variables_by_name["accu/update_accus"]
      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        # Fill accumulators.
        sess.run(tf.assign(update_accus, 1))
        m1, v1 = sess.run(
            [mean, variance],
            feed_dict={mean_in: [1.0, 2.0], variance_in: [3.0, 4.0]})
        self.assertAllClose(m1, [1.0, 2.0])
        self.assertAllClose(v1, [3.0, 4.0])
        m2, v2 = sess.run(
            [mean, variance],
            feed_dict={mean_in: [5.0, 6.0], variance_in: [7.0, 8.0]})
        self.assertAllClose(m2, [3.0, 4.0])
        self.assertAllClose(v2, [5.0, 6.0])
        # Check accumulators.
        am, av, ac = sess.run([accu_mean, accu_variance, accu_counter])
        self.assertAllClose(am, [6.0, 8.0])
        self.assertAllClose(av, [10.0, 12.0])
        self.assertAllClose([ac], [2.0])
        # Use accumulators.
        sess.run(tf.assign(update_accus, 0))
        m3, v3 = sess.run(
            [mean, variance],
            feed_dict={mean_in: [2.0, 2.0], variance_in: [3.0, 3.0]})
        self.assertAllClose(m3, [3.0, 4.0])
        self.assertAllClose(v3, [5.0, 6.0])
        am, av, ac = sess.run([accu_mean, accu_variance, accu_counter])
        self.assertAllClose(am, [6.0, 8.0])
        self.assertAllClose(av, [10.0, 12.0])
        self.assertAllClose([ac], [2.0])


if __name__ == "__main__":
  tf.test.main()
