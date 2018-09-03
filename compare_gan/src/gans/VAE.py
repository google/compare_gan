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

"""Implementation of the VAE algorithm (https://arxiv.org/abs/1312.6114)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src.gans import consts
from compare_gan.src.gans.abstract_gan import AbstractGAN
from compare_gan.src.gans.ops import lrelu, conv2d, deconv2d, batch_norm, linear, gaussian

import tensorflow as tf


class VAE(AbstractGAN):
  """Vanilla Variational Autoencoder."""

  def __init__(self, **kwargs):
    super(VAE, self).__init__("VAE", **kwargs)

  def encoder(self, x, is_training, reuse=False):
    """Implements the Gaussian Encoder."""

    sn = self.discriminator_normalization == consts.SPECTRAL_NORM
    with tf.variable_scope("encoder", reuse=reuse):
      net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name="en_conv1", use_sn=sn))
      net = conv2d(net, 128, 4, 4, 2, 2, name="en_conv2", use_sn=sn)
      if self.discriminator_normalization == consts.BATCH_NORM:
        net = batch_norm(net, is_training=is_training, scope="en_bn2")
      net = lrelu(net)
      net = tf.reshape(net, [self.batch_size, -1])
      net = linear(net, 1024, scope="en_fc3", use_sn=sn)
      if self.discriminator_normalization == consts.BATCH_NORM:
        net = batch_norm(net, is_training=is_training, scope="en_bn3")
      net = lrelu(net)

      gaussian_params = linear(net, 2 * self.z_dim, scope="en_fc4", use_sn=sn)
      mean = gaussian_params[:, :self.z_dim]
      stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])
    return mean, stddev

  def decoder(self, z, is_training, reuse=False):
    """Implements the Bernoulli decoder."""
    height = self.input_height
    width = self.input_width
    with tf.variable_scope("decoder", reuse=reuse):
      net = tf.nn.relu(
          batch_norm(
              linear(z, 1024, scope="de_fc1"),
              is_training=is_training,
              scope="de_bn1"))
      net = tf.nn.relu(
          batch_norm(
              linear(net, 128 * (height // 4) * (width // 4), scope="de_fc2"),
              is_training=is_training,
              scope="de_bn2"))
      net = tf.reshape(net, [self.batch_size, height // 4, width // 4, 128])
      net = tf.nn.relu(
          batch_norm(
              deconv2d(
                  net, [self.batch_size, height // 2, width // 2, 64],
                  4, 4, 2, 2, name="de_dc3"),
              is_training=is_training, scope="de_bn3"))
      out = tf.nn.sigmoid(
          deconv2d(net, [self.batch_size, height, width, self.c_dim],
                   4, 4, 2, 2, name="de_dc4"))
      return out

  def build_model(self, is_training=True):
    image_dims = [self.input_height, self.input_width, self.c_dim]
    batch_size = self.batch_size

    # Input images.
    self.inputs = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_images")

    # Noise vector.
    self.z = tf.placeholder(tf.float32, [batch_size, self.z_dim], name="z")

    # Encoding.
    self.mu, sigma = self.encoder(
        self.inputs, is_training=is_training, reuse=False)

    # Sammpling using the re-parameterization trick.
    z = self.mu + sigma * tf.random_normal(
        tf.shape(self.mu), 0, 1, dtype=tf.float32)

    # Decoding.
    out = self.decoder(z, is_training=is_training, reuse=False)
    self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

    # Loss function.
    marginal_likelihood = tf.reduce_sum(
        self.inputs * tf.log(self.out) +
        (1 - self.inputs) * tf.log(1 - self.out), [1, 2])
    kl_divergence = 0.5 * tf.reduce_sum(
        tf.square(self.mu) + tf.square(sigma) -
        tf.log(1e-8 + tf.square(sigma)) - 1, [1])

    self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
    self.kl_divergence = tf.reduce_mean(kl_divergence)

    self.loss = self.neg_loglikelihood + self.kl_divergence

    # Define optimization ops.
    t_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      self.optim = tf.train.AdamOptimizer(
          self.learning_rate, beta1=self.beta1).minimize(
              self.loss, var_list=t_vars)
    # Store testing images.
    self.fake_images = self.decoder(self.z, is_training=False, reuse=True)

    # Setup summaries.
    nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
    kl_sum = tf.summary.scalar("kl", self.kl_divergence)
    loss_sum = tf.summary.scalar("loss", self.loss)

    self.merged_summary_op = tf.summary.merge([nll_sum, kl_sum, loss_sum])

  def z_generator(self, batch_size, z_dim):
    return gaussian(batch_size, z_dim)

  def z_tf_generator(self, batch_size, z_dim, name=None):
    """Returns the z-generator, as tensorflow op."""
    return tf.random_normal((batch_size, z_dim), name=name)

  def run_single_train_step(self, batch_images, batch_z, counter, g_loss, sess):
    # Update the autoencoder
    _, summary_str, _, nll_loss, kl_loss = sess.run([
        self.optim, self.merged_summary_op, self.loss,
        self.neg_loglikelihood, self.kl_divergence], feed_dict={
            self.inputs: batch_images, self.z: batch_z
        })

    # Write summary.
    self.writer.add_summary(summary_str, counter)
    return kl_loss, nll_loss

  def visualize_results(self, step, sess):
    super(VAE, self).visualize_results(step, sess, z_distribution=gaussian)
