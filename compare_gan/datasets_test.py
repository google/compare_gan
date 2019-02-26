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

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
from compare_gan import datasets

import tensorflow as tf

FLAGS = flags.FLAGS

_TPU_SUPPORTED_TYPES = {
    tf.float32, tf.int32, tf.complex64, tf.int64, tf.bool, tf.bfloat16
}


def _preprocess_fn_id(images, labels):
  return {"images": images}, labels


def _preprocess_fn_add_noise(images, labels, seed=None):
  del labels
  tf.set_random_seed(seed)
  noise = tf.random.uniform([128], maxval=1.0)
  return {"images": images}, noise


class DatasetsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(DatasetsTest, self).setUp()
    FLAGS.data_shuffle_buffer_size = 100

  def get_element_and_verify_shape(self, dataset_name, expected_shape):
    dataset = datasets.get_dataset(dataset_name)
    dataset = dataset.eval_input_fn()
    image, label = dataset.make_one_shot_iterator().get_next()
    # Check if shape is known at compile time, required for TPUs.
    self.assertAllEqual(image.shape.as_list(), expected_shape)
    self.assertEqual(image.dtype, tf.float32)
    self.assertIn(label.dtype, _TPU_SUPPORTED_TYPES)
    with self.cached_session() as session:
      image = session.run(image)
      self.assertEqual(image.shape, expected_shape)
      self.assertGreaterEqual(image.min(), 0.0)
      self.assertLessEqual(image.max(), 1.0)

  def test_mnist(self):
    self.get_element_and_verify_shape("mnist", (28, 28, 1))

  def test_fashion_mnist(self):
    self.get_element_and_verify_shape("fashion-mnist", (28, 28, 1))

  def test_celeba(self):
    self.get_element_and_verify_shape("celeb_a", (64, 64, 3))

  def test_lsun(self):
    self.get_element_and_verify_shape("lsun-bedroom", (128, 128, 3))

  def _run_train_input_fn(self, dataset_name, preprocess_fn):
    dataset = datasets.get_dataset(dataset_name)
    with tf.Graph().as_default():
      dataset = dataset.input_fn(params={"batch_size": 1},
                                 preprocess_fn=preprocess_fn)
      iterator = dataset.make_initializable_iterator()
      with self.session() as sess:
        sess.run(iterator.initializer)
        next_batch = iterator.get_next()
        return [sess.run(next_batch) for _ in range(5)]

  @parameterized.named_parameters(
      ("FakeCifar", _preprocess_fn_id),
      ("FakeCifarWithRandomNoise", _preprocess_fn_add_noise),
  )
  @flagsaver.flagsaver
  def test_train_input_fn_is_determinsitic(self, preprocess_fn):
    FLAGS.data_fake_dataset = True
    batches1 = self._run_train_input_fn("cifar10", preprocess_fn)
    batches2 = self._run_train_input_fn("cifar10", preprocess_fn)
    for i in range(len(batches1)):
      # Check that both runs got the same images/noise
      self.assertAllClose(batches1[i][0], batches2[i][0])
      self.assertAllClose(batches1[i][1], batches2[i][1])

  @flagsaver.flagsaver
  def test_train_input_fn_noise_changes(self):
    FLAGS.data_fake_dataset = True
    batches = self._run_train_input_fn("cifar10", _preprocess_fn_add_noise)
    for i in range(1, len(batches)):
      self.assertNotAllClose(batches[0][1], batches[i][1])
      self.assertNotAllClose(batches[i - 1][1], batches[i][1])


if __name__ == "__main__":
  tf.test.main()
