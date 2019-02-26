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

"""Tests for the MS-SSIM score."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.metrics import ms_ssim_score
import tensorflow as tf


class MsSsimScoreTest(tf.test.TestCase):

  def test_on_one_vs_07_vs_zero_images(self):
    """Computes the SSIM value for 3 simple images."""
    with tf.Graph().as_default():
      generated_images = tf.stack([
          tf.ones([64, 64, 3]),
          tf.ones([64, 64, 3]) * 0.7,
          tf.zeros([64, 64, 3]),
      ])
      metric = ms_ssim_score.compute_msssim(generated_images, 1)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = metric(sess)
        self.assertNear(result, 0.989989, 0.001)


if __name__ == '__main__':
  tf.test.main()
