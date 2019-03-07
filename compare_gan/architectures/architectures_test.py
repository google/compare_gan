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

"""Tests for neural architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from compare_gan.architectures import dcgan
from compare_gan.architectures import infogan
from compare_gan.architectures import resnet30
from compare_gan.architectures import resnet5
from compare_gan.architectures import resnet_biggan
from compare_gan.architectures import resnet_cifar
from compare_gan.architectures import resnet_stl
from compare_gan.architectures import sndcgan
import tensorflow as tf


class ArchitectureTest(parameterized.TestCase, tf.test.TestCase):

  def assertArchitectureBuilds(self, gen, disc, image_shape, z_dim=120):
    with tf.Graph().as_default():
      batch_size = 2
      num_classes = 10
      # Prepare inputs
      z = tf.random.normal((batch_size, z_dim), name="z")
      y = tf.one_hot(tf.range(batch_size), num_classes)
      # Run check output shapes for G and D.
      x = gen(z=z, y=y, is_training=True, reuse=False)
      self.assertAllEqual(x.shape.as_list()[1:], image_shape)
      out, _, _ = disc(
          x, y=y, is_training=True, reuse=False)
      self.assertAllEqual(out.shape.as_list(), (batch_size, 1))
      # Check that G outputs valid pixel values (we use [0, 1] everywhere) and
      # D outputs a probablilty.
      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        image, pred = sess.run([x, out])
        self.assertAllGreaterEqual(image, 0)
        self.assertAllLessEqual(image, 1)
        self.assertAllGreaterEqual(pred, 0)
        self.assertAllLessEqual(pred, 1)

  @parameterized.parameters(
      {"image_shape": (28, 28, 1)},
      {"image_shape": (32, 32, 1)},
      {"image_shape": (32, 32, 3)},
      {"image_shape": (64, 64, 3)},
      {"image_shape": (128, 128, 3)},
  )
  def testDcGan(self, image_shape):
    self.assertArchitectureBuilds(
        gen=dcgan.Generator(image_shape=image_shape),
        disc=dcgan.Discriminator(),
        image_shape=image_shape)

  @parameterized.parameters(
      {"image_shape": (28, 28, 1)},
      {"image_shape": (32, 32, 1)},
      {"image_shape": (32, 32, 3)},
      {"image_shape": (64, 64, 3)},
      {"image_shape": (128, 128, 3)},
  )
  def testInfoGan(self, image_shape):
    self.assertArchitectureBuilds(
        gen=infogan.Generator(image_shape=image_shape),
        disc=infogan.Discriminator(),
        image_shape=image_shape)

  def testResNet30(self, image_shape=(128, 128, 3)):
    self.assertArchitectureBuilds(
        gen=resnet30.Generator(image_shape=image_shape),
        disc=resnet30.Discriminator(),
        image_shape=image_shape)

  @parameterized.parameters(
      {"image_shape": (32, 32, 1)},
      {"image_shape": (32, 32, 3)},
      {"image_shape": (64, 64, 3)},
      {"image_shape": (128, 128, 3)},
  )
  def testResNet5(self, image_shape):
    self.assertArchitectureBuilds(
        gen=resnet5.Generator(image_shape=image_shape),
        disc=resnet5.Discriminator(),
        image_shape=image_shape)

  @parameterized.parameters(
      {"image_shape": (32, 32, 3)},
      {"image_shape": (64, 64, 3)},
      {"image_shape": (128, 128, 3)},
      {"image_shape": (256, 256, 3)},
      {"image_shape": (512, 512, 3)},
  )
  def testResNet5BigGan(self, image_shape):
    if image_shape[0] == 512:
      z_dim = 160
    elif image_shape[0] == 256:
      z_dim = 140
    else:
      z_dim = 120
    # Use channel multiplier 4 to avoid OOM errors.
    self.assertArchitectureBuilds(
        gen=resnet_biggan.Generator(image_shape=image_shape, ch=16),
        disc=resnet_biggan.Discriminator(ch=16),
        image_shape=image_shape,
        z_dim=z_dim)

  @parameterized.parameters(
      {"image_shape": (32, 32, 1)},
      {"image_shape": (32, 32, 3)},
  )
  def testResNetCifar(self, image_shape):
    self.assertArchitectureBuilds(
        gen=resnet_cifar.Generator(image_shape=image_shape),
        disc=resnet_cifar.Discriminator(),
        image_shape=image_shape)

  @parameterized.parameters(
      {"image_shape": (48, 48, 1)},
      {"image_shape": (48, 48, 3)},
  )
  def testResNetStl(self, image_shape):
    self.assertArchitectureBuilds(
        gen=resnet_stl.Generator(image_shape=image_shape),
        disc=resnet_stl.Discriminator(),
        image_shape=image_shape)

  @parameterized.parameters(
      {"image_shape": (28, 28, 1)},
      {"image_shape": (32, 32, 1)},
      {"image_shape": (32, 32, 3)},
      {"image_shape": (64, 64, 3)},
      {"image_shape": (128, 128, 3)},
  )
  def testSnDcGan(self, image_shape):
    self.assertArchitectureBuilds(
        gen=sndcgan.Generator(image_shape=image_shape),
        disc=sndcgan.Discriminator(),
        image_shape=image_shape)


if __name__ == "__main__":
  tf.test.main()
