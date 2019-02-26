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

"""Tests weight initialization ops using ResNet5 architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import resnet5
from compare_gan.gans import consts
import gin
import tensorflow as tf


class ResNetInitTest(tf.test.TestCase):

  def setUp(self):
    super(ResNetInitTest, self).setUp()
    gin.clear_config()

  def testInitializersOldDefault(self):
    valid_initalizer = [
        "kernel/Initializer/random_normal",
        "bias/Initializer/Const",
        # truncated_normal is the old default for conv2d.
        "kernel/Initializer/truncated_normal",
        "bias/Initializer/Const",
        "beta/Initializer/zeros",
        "gamma/Initializer/ones",
    ]
    valid_op_names = "/({}):0$".format("|".join(valid_initalizer))
    with tf.Graph().as_default():
      z = tf.zeros((2, 128))
      fake_image = resnet5.Generator(image_shape=(128, 128, 3))(
          z, y=None, is_training=True)
      resnet5.Discriminator()(fake_image, y=None, is_training=True)
      for var in tf.trainable_variables():
        op_name = var.initializer.inputs[1].name
        self.assertRegex(op_name, valid_op_names)

  def testInitializersRandomNormal(self):
    gin.bind_parameter("weights.initializer", consts.NORMAL_INIT)
    valid_initalizer = [
        "kernel/Initializer/random_normal",
        "bias/Initializer/Const",
        "kernel/Initializer/random_normal",
        "bias/Initializer/Const",
        "beta/Initializer/zeros",
        "gamma/Initializer/ones",
    ]
    valid_op_names = "/({}):0$".format("|".join(valid_initalizer))
    with tf.Graph().as_default():
      z = tf.zeros((2, 128))
      fake_image = resnet5.Generator(image_shape=(128, 128, 3))(
          z, y=None, is_training=True)
      resnet5.Discriminator()(fake_image, y=None, is_training=True)
      for var in tf.trainable_variables():
        op_name = var.initializer.inputs[1].name
        self.assertRegex(op_name, valid_op_names)

  def testInitializersTruncatedNormal(self):
    gin.bind_parameter("weights.initializer", consts.TRUNCATED_INIT)
    valid_initalizer = [
        "kernel/Initializer/truncated_normal",
        "bias/Initializer/Const",
        "kernel/Initializer/truncated_normal",
        "bias/Initializer/Const",
        "beta/Initializer/zeros",
        "gamma/Initializer/ones",
    ]
    valid_op_names = "/({}):0$".format("|".join(valid_initalizer))
    with tf.Graph().as_default():
      z = tf.zeros((2, 128))
      fake_image = resnet5.Generator(image_shape=(128, 128, 3))(
          z, y=None, is_training=True)
      resnet5.Discriminator()(fake_image, y=None, is_training=True)
      for var in tf.trainable_variables():
        op_name = var.initializer.inputs[1].name
        self.assertRegex(op_name, valid_op_names)

  def testGeneratorInitializersOrthogonal(self):
    gin.bind_parameter("weights.initializer", consts.ORTHOGONAL_INIT)
    valid_initalizer = [
        "kernel/Initializer/mul_1",
        "bias/Initializer/Const",
        "kernel/Initializer/mul_1",
        "bias/Initializer/Const",
        "beta/Initializer/zeros",
        "gamma/Initializer/ones",
    ]
    valid_op_names = "/({}):0$".format("|".join(valid_initalizer))
    with tf.Graph().as_default():
      z = tf.zeros((2, 128))
      fake_image = resnet5.Generator(image_shape=(128, 128, 3))(
          z, y=None, is_training=True)
      resnet5.Discriminator()(fake_image, y=None, is_training=True)
      for var in tf.trainable_variables():
        op_name = var.initializer.inputs[1].name
        self.assertRegex(op_name, valid_op_names)


if __name__ == "__main__":
  tf.test.main()
