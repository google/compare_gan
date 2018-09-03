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

"""Tests for Resnet architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src.gans import resnet_architecture as resnet_arch

import tensorflow as tf


def TestResnet5GeneratorShape(output_shape):
  config = tf.ConfigProto(allow_soft_placement=True)
  tf.reset_default_graph()
  batch_size = 8
  z_dim = 64
  os = output_shape
  with tf.Session(config=config) as sess:
    z = tf.random_normal([batch_size, z_dim])
    g = resnet_arch.resnet5_generator(
        noise=z, is_training=True, reuse=False, colors=3, output_shape=os)
    tf.global_variables_initializer().run()
    output = sess.run([g])
    return [output[0].shape, (batch_size, os, os, 3)]


class ResnetArchitectureTest(tf.test.TestCase):

  def testResnet5GeneratorRuns(self):
    generator_128 = TestResnet5GeneratorShape(128)
    generator_64 = TestResnet5GeneratorShape(64)
    self.assertEquals(generator_128[0], generator_128[1])
    self.assertEquals(generator_64[0], generator_64[1])

  def testResnet5DiscriminatorRuns(self):
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.reset_default_graph()
    batch_size = 8
    with tf.Session(config=config) as sess:
      images = tf.random_normal([batch_size, 128, 128, 3])
      out, _, _ = resnet_arch.resnet5_discriminator(
          images, is_training=True, discriminator_normalization="spectral_norm",
          reuse=False)
      tf.global_variables_initializer().run()
      output = sess.run([out])
      self.assertEquals(output[0].shape, (batch_size, 1))

  def testResnet107GeneratorRuns(self):
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.reset_default_graph()
    batch_size = 8
    z_dim = 64
    with tf.Session(config=config) as sess:
      z = tf.random_normal([batch_size, z_dim])
      g = resnet_arch.resnet107_generator(
          noise=z, is_training=True, reuse=False, colors=3)
      tf.global_variables_initializer().run()
      output = sess.run([g])
      self.assertEquals(output[0].shape, (batch_size, 128, 128, 3))

  def testResnet107DiscriminatorRuns(self):
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.reset_default_graph()
    batch_size = 8
    with tf.Session(config=config) as sess:
      images = tf.random_normal([batch_size, 128, 128, 3])
      out, _, _ = resnet_arch.resnet107_discriminator(
          images, is_training=True,
          discriminator_normalization="spectral_norm", reuse=False)
      tf.global_variables_initializer().run()
      output = sess.run([out])
      self.assertEquals(output[0].shape, (batch_size, 1))

if __name__ == "__main__":
  tf.test.main()
