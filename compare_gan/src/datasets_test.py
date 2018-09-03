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

"""Tests for datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from compare_gan.src import datasets

import tensorflow as tf

FLAGS = tf.flags.FLAGS

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, "test_data")


class DatasetsTest(tf.test.TestCase):

  def setUp(self):
    FLAGS.dataset_root = _TESTDATA

  def test_just_create_dataset(self):
    datasets.load_fake("fake", "train", 1, 10)
    datasets.load_mnist("mnist", "train", 1, 10)
    datasets.load_fashion_mnist("fashion_mnist", "train", 1, 10)
    datasets.load_cifar10("cifar10", "train", 1, 10)
    datasets.load_celeba("celeba", "train", 1, 10)
    datasets.load_lsun("lsun", "train", 1, 10)
    datasets.load_celebahq("celebahq128", "train", 1, 10)

  def get_element_and_verify_shape(self, dataset, expected_shape):
    element = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as session:
      image, _ = session.run(element)
      self.assertEqual(image.shape, expected_shape)
      self.assertGreaterEqual(image.min(), 0.0)
      self.assertLessEqual(image.max(), 1.0)

  def test_mnist(self):
    self.get_element_and_verify_shape(
        datasets.load_mnist("mnist", "dev", 1, 10),
        (28, 28, 1))

  def test_fashion_mnist(self):
    self.get_element_and_verify_shape(
        datasets.load_fashion_mnist("fashion_mnist", "dev", 1, 10),
        (28, 28, 1))

  def test_cifar10(self):
    self.get_element_and_verify_shape(
        datasets.load_cifar10("cifar10", "dev", 1, 10),
        (32, 32, 3))

  def test_celeba(self):
    self.get_element_and_verify_shape(
        datasets.load_celeba("celeba", "dev", 1, 10),
        (64, 64, 3))

  def test_lsun(self):
    self.get_element_and_verify_shape(
        datasets.load_lsun("lsun", "dev", 1, 10),
        (128, 128, 3))

  def test_celebahq(self):
    self.get_element_and_verify_shape(
        datasets.load_celebahq("celebahq128", "dev", 1, 10),
        (128, 128, 3))


if __name__ == "__main__":
  tf.test.main()
