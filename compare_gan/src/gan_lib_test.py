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

"""Tests for compare_gan.gan_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src import gan_lib

import tensorflow as tf


class GanLibTest(tf.test.TestCase):

  def testLoadingTriangles(self):
    with tf.Graph().as_default():
      iterator = gan_lib.load_dataset("triangles").batch(
          32).make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (32, 28, 28, 1))
        self.assertEqual(label.shape, (32,))
        self.assertEqual(label[4], 3)
    with tf.Graph().as_default():
      iterator = gan_lib.load_dataset(
          "triangles", split_name="test").make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (28, 28, 1))
        self.assertEqual(label.shape, ())
    with tf.Graph().as_default():
      iterator = gan_lib.load_dataset(
          "triangles", split_name="validation").make_one_shot_iterator(
              ).get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (28, 28, 1))
        self.assertEqual(label.shape, ())

  def testLoadingMnist(self):
    with tf.Graph().as_default():
      dataset = gan_lib.load_dataset("mnist")
      iterator = dataset.make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (28, 28, 1))
        self.assertEqual(label.shape, ())

if __name__ == "__main__":
  tf.test.main()
