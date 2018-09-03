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

"""Implementation of the DRAGAN algorithm (https://arxiv.org/abs/1705.07215)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src.gans.abstract_gan import AbstractGAN

import numpy as np
import tensorflow as tf


class DRAGAN(AbstractGAN):
  """How to Train Your DRAGAN."""

  def __init__(self, **kwargs):
    super(DRAGAN, self).__init__("DRAGAN", **kwargs)

    # Higher value: more stable, but slower convergence.
    self.lambd = self.parameters["lambda"]

  def get_perturbed_batch(self, minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

  def build_model(self, is_training=True):
    image_dims = [self.input_height, self.input_width, self.c_dim]
    batch_size = self.batch_size

    # Input images.
    self.inputs = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_images")
    self.inputs_p = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_perturbed_images")

    # Noise vector.
    self.z = tf.placeholder(tf.float32, [batch_size, self.z_dim], name="z")

    # Discriminator output for real images.
    D_real, D_real_logits, _ = self.discriminator(
        self.inputs, is_training=is_training, reuse=False)

    # Discriminator output for fake images.
    G = self.generator(self.z, is_training=is_training, reuse=False)
    D_fake, D_fake_logits, _ = self.discriminator(
        G, is_training=is_training, reuse=True)

    # Loss on real and fake data.
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_real_logits, labels=tf.ones_like(D_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

    # Total discriminator loss.
    self.d_loss = d_loss_real + d_loss_fake

    # Total generator loss.
    self.g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake_logits, labels=tf.ones_like(D_fake)))

    # Gradient penalty.
    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = self.inputs_p - self.inputs
    interpolates = self.inputs + (alpha * differences)
    D_inter, _, _ = self.discriminator(
        interpolates, is_training=is_training, reuse=True)
    gradients = tf.gradients(D_inter, [interpolates])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    self.d_loss += self.lambd * gradient_penalty

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

  def discriminator_feed_dict(self, features, labels):
    del labels
    return {
        self.inputs: features["images"],
        self.inputs_p: self.get_perturbed_batch(features["images"]),
        self.z: features["z_for_disc_step"],
    }
