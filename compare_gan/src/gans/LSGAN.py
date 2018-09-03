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

"""Implementation of the LS-GAN algorithm (https://arxiv.org/abs/1611.04076)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src.gans.abstract_gan import AbstractGAN

import tensorflow as tf


class LSGAN(AbstractGAN):
  """Least Squares Generative Adversarial Networks."""

  def __init__(self, **kwargs):
    super(LSGAN, self).__init__("LSGAN", **kwargs)

  def build_model(self, is_training=True):
    image_dims = [self.input_height, self.input_width, self.c_dim]
    batch_size = self.batch_size

    # Input images.
    self.inputs = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_images")

    # Noise vector.
    self.z = tf.placeholder(tf.float32, [batch_size, self.z_dim], name="z")

    # Discriminator output for real images.
    D_real, D_real_logits, _ = self.discriminator(
        self.inputs, is_training=is_training, reuse=False)

    # Discriminator output for fake images.
    G = self.generator(self.z, is_training=is_training, reuse=False)
    D_fake, D_fake_logits, _ = self.discriminator(
        G, is_training=is_training, reuse=True)

    # Total discriminator loss.
    D_real = tf.squeeze(D_real)
    D_fake = tf.squeeze(D_fake)
    assert D_real.get_shape() == (batch_size,)
    assert D_fake.get_shape() == (batch_size,)
    d_loss_real = tf.reduce_mean(tf.square(D_real - 1.0))
    d_loss_fake = tf.reduce_mean(tf.square(D_fake))
    self.d_loss = 0.5 * (d_loss_real + d_loss_fake)

    # Total generator loss.
    self.g_loss = 0.5 * tf.reduce_mean(tf.square(D_fake - 1))

    # Divide trainable variables into a group for D and group for G.
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    g_vars = [var for var in t_vars if "generator" in var.name]
    self.check_variables(t_vars, d_vars, g_vars)

    # Define optimization ops.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      self.d_optim = tf.train.AdamOptimizer(
          self.learning_rate, beta1=self.beta1, name="d_adam").minimize(
              self.d_loss, var_list=d_vars)
      self.g_optim = tf.train.AdamOptimizer(
          self.learning_rate, beta1=self.beta1, name="g_adam").minimize(
              self.g_loss, var_list=g_vars)

    # Store testing images.
    self.fake_images = self.generator(self.z, is_training=False, reuse=True)

    # Setup summaries.
    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

    self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
    self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
