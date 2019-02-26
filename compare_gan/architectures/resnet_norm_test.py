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

"""Tests batch normalizations using ResNet5 CIFAR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from compare_gan.architectures import arch_ops
from compare_gan.architectures import resnet_cifar
from six.moves import zip
import tensorflow as tf


class ResNetNormTest(tf.test.TestCase):

  def testDefaultGenerator(self):
    with tf.Graph().as_default():
      # batch size 8, 32x32x3 images, 10 classes.
      z = tf.zeros((8, 128))
      y = tf.one_hot(tf.ones((8,), dtype=tf.int32), 10)
      generator = resnet_cifar.Generator(image_shape=(32, 32, 3))
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      self.assertEqual(fake_images.shape.as_list(), [8, 32, 32, 3])
      expected_variables = [
          # Name and shape.
          ("generator/fc_noise/kernel:0", [128, 4096]),
          ("generator/fc_noise/bias:0", [4096]),
          ("generator/B1/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv_shortcut/bias:0", [256]),
          ("generator/B1/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv1/bias:0", [256]),
          ("generator/B1/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/same_conv2/bias:0", [256]),
          ("generator/B2/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv_shortcut/bias:0", [256]),
          ("generator/B2/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv1/bias:0", [256]),
          ("generator/B2/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/same_conv2/bias:0", [256]),
          ("generator/B3/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv_shortcut/bias:0", [256]),
          ("generator/B3/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv1/bias:0", [256]),
          ("generator/B3/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/same_conv2/bias:0", [256]),
          ("generator/final_conv/kernel:0", [3, 3, 256, 3]),
          ("generator/final_conv/bias:0", [3]),
      ]
      actual_variables = [(v.name, v.shape.as_list())
                          for v in tf.trainable_variables()]
      for a, e in zip(actual_variables, expected_variables):
        logging.info("actual: %s, expected: %s", a, e)
        self.assertEqual(a, e)
      self.assertEqual(len(actual_variables), len(expected_variables))

  def testDefaultDiscriminator(self):
    with tf.Graph().as_default():
      # batch size 8, 32x32x3 images, 10 classes.
      x = tf.zeros((8, 32, 32, 3))
      y = tf.one_hot(tf.ones((8,), dtype=tf.int32), 10)
      discriminator = resnet_cifar.Discriminator()
      _ = discriminator(x, y=y, is_training=True, reuse=False)
      expected_variables = [
          # Name and shape.
          ("discriminator/B1/down_conv_shortcut/kernel:0", [3, 3, 3, 128]),
          ("discriminator/B1/down_conv_shortcut/bias:0", [128]),
          ("discriminator/B1/same_conv1/kernel:0", [3, 3, 3, 128]),
          ("discriminator/B1/same_conv1/bias:0", [128]),
          ("discriminator/B1/down_conv2/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B1/down_conv2/bias:0", [128]),
          ("discriminator/B2/down_conv_shortcut/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B2/down_conv_shortcut/bias:0", [128]),
          ("discriminator/B2/same_conv1/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B2/same_conv1/bias:0", [128]),
          ("discriminator/B2/down_conv2/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B2/down_conv2/bias:0", [128]),
          ("discriminator/B3/same_conv_shortcut/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B3/same_conv_shortcut/bias:0", [128]),
          ("discriminator/B3/same_conv1/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B3/same_conv1/bias:0", [128]),
          ("discriminator/B3/same_conv2/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B3/same_conv2/bias:0", [128]),
          ("discriminator/B4/same_conv_shortcut/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B4/same_conv_shortcut/bias:0", [128]),
          ("discriminator/B4/same_conv1/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B4/same_conv1/bias:0", [128]),
          ("discriminator/B4/same_conv2/kernel:0", [3, 3, 128, 128]),
          ("discriminator/B4/same_conv2/bias:0", [128]),
          ("discriminator/disc_final_fc/kernel:0", [128, 1]),
          ("discriminator/disc_final_fc/bias:0", [1]),
      ]
      actual_variables = [(v.name, v.shape.as_list())
                          for v in tf.trainable_variables()]
      for a, e in zip(actual_variables, expected_variables):
        logging.info("actual: %s, expected: %s", a, e)
        self.assertEqual(a, e)
      self.assertEqual(len(actual_variables), len(expected_variables))

  def testDefaultGeneratorWithBatchNorm(self):
    with tf.Graph().as_default():
      # batch size 8, 32x32x3 images, 10 classes.
      z = tf.zeros((8, 128))
      y = tf.one_hot(tf.ones((8,), dtype=tf.int32), 10)
      generator = resnet_cifar.Generator(
          image_shape=(32, 32, 3),
          batch_norm_fn=arch_ops.batch_norm)
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      self.assertEqual(fake_images.shape.as_list(), [8, 32, 32, 3])
      expected_variables = [
          # Name and shape.
          ("generator/fc_noise/kernel:0", [128, 4096]),
          ("generator/fc_noise/bias:0", [4096]),
          ("generator/B1/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv_shortcut/bias:0", [256]),
          ("generator/B1/bn1/gamma:0", [256]),
          ("generator/B1/bn1/beta:0", [256]),
          ("generator/B1/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv1/bias:0", [256]),
          ("generator/B1/bn2/gamma:0", [256]),
          ("generator/B1/bn2/beta:0", [256]),
          ("generator/B1/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/same_conv2/bias:0", [256]),
          ("generator/B2/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv_shortcut/bias:0", [256]),
          ("generator/B2/bn1/gamma:0", [256]),
          ("generator/B2/bn1/beta:0", [256]),
          ("generator/B2/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv1/bias:0", [256]),
          ("generator/B2/bn2/gamma:0", [256]),
          ("generator/B2/bn2/beta:0", [256]),
          ("generator/B2/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/same_conv2/bias:0", [256]),
          ("generator/B3/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv_shortcut/bias:0", [256]),
          ("generator/B3/bn1/gamma:0", [256]),
          ("generator/B3/bn1/beta:0", [256]),
          ("generator/B3/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv1/bias:0", [256]),
          ("generator/B3/bn2/gamma:0", [256]),
          ("generator/B3/bn2/beta:0", [256]),
          ("generator/B3/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/same_conv2/bias:0", [256]),
          ("generator/final_norm/gamma:0", [256]),
          ("generator/final_norm/beta:0", [256]),
          ("generator/final_conv/kernel:0", [3, 3, 256, 3]),
          ("generator/final_conv/bias:0", [3]),
      ]
      actual_variables = [(v.name, v.shape.as_list())
                          for v in tf.trainable_variables()]
      for a in actual_variables:
        logging.info(a)
      for a, e in zip(actual_variables, expected_variables):
        logging.info("actual: %s, expected: %s", a, e)
        self.assertEqual(a, e)
      self.assertEqual(len(actual_variables), len(expected_variables))

  def testDefaultGeneratorWithConditionalBatchNorm(self):
    with tf.Graph().as_default():
      # Batch size 8, 32x32x3 images, 10 classes.
      z = tf.zeros((8, 128))
      y = tf.one_hot(tf.ones((8,), dtype=tf.int32), 10)
      generator = resnet_cifar.Generator(
          image_shape=(32, 32, 3),
          batch_norm_fn=arch_ops.conditional_batch_norm)
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      self.assertEqual(fake_images.shape.as_list(), [8, 32, 32, 3])
      expected_variables = [
          # Name and shape.
          ("generator/fc_noise/kernel:0", [128, 4096]),
          ("generator/fc_noise/bias:0", [4096]),
          ("generator/B1/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv_shortcut/bias:0", [256]),
          ("generator/B1/bn1/condition/gamma/kernel:0", [10, 256]),
          ("generator/B1/bn1/condition/beta/kernel:0", [10, 256]),
          ("generator/B1/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv1/bias:0", [256]),
          ("generator/B1/bn2/condition/gamma/kernel:0", [10, 256]),
          ("generator/B1/bn2/condition/beta/kernel:0", [10, 256]),
          ("generator/B1/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/same_conv2/bias:0", [256]),
          ("generator/B2/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv_shortcut/bias:0", [256]),
          ("generator/B2/bn1/condition/gamma/kernel:0", [10, 256]),
          ("generator/B2/bn1/condition/beta/kernel:0", [10, 256]),
          ("generator/B2/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv1/bias:0", [256]),
          ("generator/B2/bn2/condition/gamma/kernel:0", [10, 256]),
          ("generator/B2/bn2/condition/beta/kernel:0", [10, 256]),
          ("generator/B2/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/same_conv2/bias:0", [256]),
          ("generator/B3/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv_shortcut/bias:0", [256]),
          ("generator/B3/bn1/condition/gamma/kernel:0", [10, 256]),
          ("generator/B3/bn1/condition/beta/kernel:0", [10, 256]),
          ("generator/B3/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv1/bias:0", [256]),
          ("generator/B3/bn2/condition/gamma/kernel:0", [10, 256]),
          ("generator/B3/bn2/condition/beta/kernel:0", [10, 256]),
          ("generator/B3/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/same_conv2/bias:0", [256]),
          ("generator/final_norm/condition/gamma/kernel:0", [10, 256]),
          ("generator/final_norm/condition/beta/kernel:0", [10, 256]),
          ("generator/final_conv/kernel:0", [3, 3, 256, 3]),
          ("generator/final_conv/bias:0", [3]),
      ]
      actual_variables = [(v.name, v.shape.as_list())
                          for v in tf.trainable_variables()]
      for a in actual_variables:
        logging.info(a)
      for a, e in zip(actual_variables, expected_variables):
        logging.info("actual: %s, expected: %s", a, e)
        self.assertEqual(a, e)
      self.assertEqual(len(actual_variables), len(expected_variables))

  def testDefaultGeneratorWithSelfModulatedBatchNorm(self):
    with tf.Graph().as_default():
      # Batch size 8, 32x32x3 images, 10 classes.
      z = tf.zeros((8, 128))
      y = tf.one_hot(tf.ones((8,), dtype=tf.int32), 10)
      generator = resnet_cifar.Generator(
          image_shape=(32, 32, 3),
          batch_norm_fn=arch_ops.self_modulated_batch_norm)
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      self.assertEqual(fake_images.shape.as_list(), [8, 32, 32, 3])
      expected_variables = [
          # Name and shape.
          ("generator/fc_noise/kernel:0", [128, 4096]),
          ("generator/fc_noise/bias:0", [4096]),
          ("generator/B1/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv_shortcut/bias:0", [256]),
          ("generator/B1/bn1/sbn/hidden/kernel:0", [128, 32]),
          ("generator/B1/bn1/sbn/hidden/bias:0", [32]),
          ("generator/B1/bn1/sbn/gamma/kernel:0", [32, 256]),
          ("generator/B1/bn1/sbn/gamma/bias:0", [256]),
          ("generator/B1/bn1/sbn/beta/kernel:0", [32, 256]),
          ("generator/B1/bn1/sbn/beta/bias:0", [256]),
          ("generator/B1/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv1/bias:0", [256]),
          ("generator/B1/bn2/sbn/hidden/kernel:0", [128, 32]),
          ("generator/B1/bn2/sbn/hidden/bias:0", [32]),
          ("generator/B1/bn2/sbn/gamma/kernel:0", [32, 256]),
          ("generator/B1/bn2/sbn/gamma/bias:0", [256]),
          ("generator/B1/bn2/sbn/beta/kernel:0", [32, 256]),
          ("generator/B1/bn2/sbn/beta/bias:0", [256]),
          ("generator/B1/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/same_conv2/bias:0", [256]),
          ("generator/B2/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv_shortcut/bias:0", [256]),
          ("generator/B2/bn1/sbn/hidden/kernel:0", [128, 32]),
          ("generator/B2/bn1/sbn/hidden/bias:0", [32]),
          ("generator/B2/bn1/sbn/gamma/kernel:0", [32, 256]),
          ("generator/B2/bn1/sbn/gamma/bias:0", [256]),
          ("generator/B2/bn1/sbn/beta/kernel:0", [32, 256]),
          ("generator/B2/bn1/sbn/beta/bias:0", [256]),
          ("generator/B2/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv1/bias:0", [256]),
          ("generator/B2/bn2/sbn/hidden/kernel:0", [128, 32]),
          ("generator/B2/bn2/sbn/hidden/bias:0", [32]),
          ("generator/B2/bn2/sbn/gamma/kernel:0", [32, 256]),
          ("generator/B2/bn2/sbn/gamma/bias:0", [256]),
          ("generator/B2/bn2/sbn/beta/kernel:0", [32, 256]),
          ("generator/B2/bn2/sbn/beta/bias:0", [256]),
          ("generator/B2/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/same_conv2/bias:0", [256]),
          ("generator/B3/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv_shortcut/bias:0", [256]),
          ("generator/B3/bn1/sbn/hidden/kernel:0", [128, 32]),
          ("generator/B3/bn1/sbn/hidden/bias:0", [32]),
          ("generator/B3/bn1/sbn/gamma/kernel:0", [32, 256]),
          ("generator/B3/bn1/sbn/gamma/bias:0", [256]),
          ("generator/B3/bn1/sbn/beta/kernel:0", [32, 256]),
          ("generator/B3/bn1/sbn/beta/bias:0", [256]),
          ("generator/B3/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv1/bias:0", [256]),
          ("generator/B3/bn2/sbn/hidden/kernel:0", [128, 32]),
          ("generator/B3/bn2/sbn/hidden/bias:0", [32]),
          ("generator/B3/bn2/sbn/gamma/kernel:0", [32, 256]),
          ("generator/B3/bn2/sbn/gamma/bias:0", [256]),
          ("generator/B3/bn2/sbn/beta/kernel:0", [32, 256]),
          ("generator/B3/bn2/sbn/beta/bias:0", [256]),
          ("generator/B3/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/same_conv2/bias:0", [256]),
          ("generator/final_norm/sbn/hidden/kernel:0", [128, 32]),
          ("generator/final_norm/sbn/hidden/bias:0", [32]),
          ("generator/final_norm/sbn/gamma/kernel:0", [32, 256]),
          ("generator/final_norm/sbn/gamma/bias:0", [256]),
          ("generator/final_norm/sbn/beta/kernel:0", [32, 256]),
          ("generator/final_norm/sbn/beta/bias:0", [256]),
          ("generator/final_conv/kernel:0", [3, 3, 256, 3]),
          ("generator/final_conv/bias:0", [3]),
      ]
      actual_variables = [(v.name, v.shape.as_list())
                          for v in tf.trainable_variables()]
      for a in actual_variables:
        logging.info(a)
      for a, e in zip(actual_variables, expected_variables):
        logging.info("actual: %s, expected: %s", a, e)
        self.assertEqual(a, e)
      self.assertEqual(len(actual_variables), len(expected_variables))

  def testDefaultGeneratorWithSpectralNorm(self):
    with tf.Graph().as_default():
      # Batch size 8, 32x32x3 images, 10 classes.
      z = tf.zeros((8, 128))
      y = tf.one_hot(tf.ones((8,), dtype=tf.int32), 10)
      generator = resnet_cifar.Generator(
          image_shape=(32, 32, 3),
          spectral_norm=True)
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      self.assertEqual(fake_images.shape.as_list(), [8, 32, 32, 3])
      expected_variables = [
          # Name and shape.
          ("generator/fc_noise/kernel:0", [128, 4096]),
          ("generator/fc_noise/kernel/u_var:0", [128, 1]),
          ("generator/fc_noise/bias:0", [4096]),
          ("generator/B1/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv_shortcut/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B1/up_conv_shortcut/bias:0", [256]),
          ("generator/B1/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/up_conv1/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B1/up_conv1/bias:0", [256]),
          ("generator/B1/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B1/same_conv2/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B1/same_conv2/bias:0", [256]),
          ("generator/B2/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv_shortcut/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B2/up_conv_shortcut/bias:0", [256]),
          ("generator/B2/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/up_conv1/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B2/up_conv1/bias:0", [256]),
          ("generator/B2/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B2/same_conv2/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B2/same_conv2/bias:0", [256]),
          ("generator/B3/up_conv_shortcut/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv_shortcut/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B3/up_conv_shortcut/bias:0", [256]),
          ("generator/B3/up_conv1/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/up_conv1/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B3/up_conv1/bias:0", [256]),
          ("generator/B3/same_conv2/kernel:0", [3, 3, 256, 256]),
          ("generator/B3/same_conv2/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/B3/same_conv2/bias:0", [256]),
          ("generator/final_conv/kernel:0", [3, 3, 256, 3]),
          ("generator/final_conv/kernel/u_var:0", [3 * 3 * 256, 1]),
          ("generator/final_conv/bias:0", [3]),
      ]
      actual_variables = [(v.name, v.shape.as_list())
                          for v in tf.global_variables()]
      for a in actual_variables:
        logging.info(a)
      for a, e in zip(actual_variables, expected_variables):
        logging.info("actual: %s, expected: %s", a, e)
        self.assertEqual(a, e)
      self.assertEqual(len(actual_variables), len(expected_variables))


if __name__ == "__main__":
  tf.test.main()
