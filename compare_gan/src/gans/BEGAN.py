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

"""Implementation of the BEGAN algorithm (https://arxiv.org/abs/1703.10717)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src.gans import consts
from compare_gan.src.gans.abstract_gan import AbstractGAN
from compare_gan.src.gans.ops import batch_norm, linear, conv2d, deconv2d, lrelu

import tensorflow as tf


class BEGAN(AbstractGAN):
  """Boundary Equilibrium Generative Adversarial Networks."""

  def __init__(self, **kwargs):
    super(BEGAN, self).__init__("BEGAN", **kwargs)

    self.gamma = self.parameters["gamma"]
    self.lambd = self.parameters["lambda"]

  def discriminator(self, x, is_training, reuse=False):
    """BEGAN discriminator (auto-encoder).

       This implementation doesn't match the one from the paper, but is similar
       to our "standard" discriminator (same 2 conv layers, using lrelu).
       However, it still has less parameters (1.3M vs 8.5M) because of the huge
       linear layer in the standard discriminator.

    Args:
      x: input images, shape [bs, h, w, channels]
      is_training: boolean, are we in train or eval model.
      reuse: boolean, should params be re-used.

    Returns:
      out: a float (in [0, 1]) with discriminator prediction
      recon_error: L1 reconstrunction error of the auto-encoder
      code: the representation (bottleneck layer of the auto-encoder)
    """
    height = self.input_height
    width = self.input_width
    sn = self.discriminator_normalization == consts.SPECTRAL_NORM
    with tf.variable_scope("discriminator", reuse=reuse):
      # Encoding step (Mapping from [bs, h, w, c] to [bs, 64])
      net = conv2d(
          x, 64, 4, 4, 2, 2, name="d_conv1", use_sn=sn)  # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = conv2d(
          net, 128, 4, 4, 2, 2, name="d_conv2",
          use_sn=sn)  # [bs, h/4, w/4, 128]
      net = tf.reshape(net, [self.batch_size, -1])  # [bs, h * w * 8]
      code = linear(net, 64, scope="d_fc6", use_sn=sn)  # [bs, 64]
      if self.discriminator_normalization == consts.BATCH_NORM:
        code = batch_norm(code, is_training=is_training, scope="d_bn1")
      code = lrelu(code)

      # Decoding step (Mapping from [bs, 64] to [bs, h, w, c])
      net = linear(
          code, 128 * (height // 4) * (width // 4), scope="d_fc1",
          use_sn=sn)  # [bs, h/4 * w/4 * 128]
      if self.discriminator_normalization == consts.BATCH_NORM:
        net = batch_norm(net, is_training=is_training, scope="d_bn2")
      net = lrelu(net)
      net = tf.reshape(net, [
          self.batch_size, height // 4, width // 4, 128])  # [bs, h/4, w/4, 128]
      net = deconv2d(net, [self.batch_size, height // 2, width // 2, 64],
                     4, 4, 2, 2, name="d_deconv1")  # [bs, h/2, w/2, 64]
      if self.discriminator_normalization == consts.BATCH_NORM:
        net = batch_norm(net, is_training=is_training, scope="d_bn3")
      net = lrelu(net)
      net = deconv2d(net, [self.batch_size, height, width, self.c_dim],
                     4, 4, 2, 2, name="d_deconv2")  # [bs, h, w, c]
      out = tf.nn.sigmoid(net)

      # Reconstruction loss.
      recon_error = tf.reduce_mean(tf.abs(out - x))
      return out, recon_error, code

  def build_model(self, is_training=True):
    image_dims = [self.input_height, self.input_width, self.c_dim]
    batch_size = self.batch_size

    # BEGAN parameter.
    self.k = tf.Variable(0., trainable=False)

    # Input images.
    self.inputs = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_images")
    # Noise vector.
    self.z = tf.placeholder(tf.float32, [batch_size, self.z_dim], name="z")

    # Discriminator loss for real images.
    D_real_img, D_real_err, D_real_code = self.discriminator(
        self.inputs, is_training=is_training, reuse=False)

    # Discriminator loss for fake images.
    G = self.generator(self.z, is_training=is_training, reuse=False)
    D_fake_img, D_fake_err, D_fake_code = self.discriminator(
        G, is_training=is_training, reuse=True)

    # Total discriminator loss.
    self.d_loss = D_real_err - self.k * D_fake_err

    # Total generator loss.
    self.g_loss = D_fake_err

    # Convergence metric.
    self.M = D_real_err + tf.abs(self.gamma * D_real_err - D_fake_err)

    # Op for updating k.
    self.update_k = self.k.assign(self.k + self.lambd *
                                  (self.gamma * D_real_err - D_fake_err))

    # Divide trainable variables into a group for D and a group for G.
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
    d_loss_real_sum = tf.summary.scalar("d_error_real", D_real_err)
    d_loss_fake_sum = tf.summary.scalar("d_error_fake", D_fake_err)
    d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    M_sum = tf.summary.scalar("M", self.M)
    k_sum = tf.summary.scalar("k", self.k)

    self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
    self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
    self.p_sum = tf.summary.merge([M_sum, k_sum])

  def after_training_step_hook(self, sess, features, counter):
    batch_images = features["images"]
    batch_z = features["z_for_disc_step"]
    # Update k.
    summary_str = sess.run(
        [self.update_k, self.p_sum, self.M, self.k],
        feed_dict={
            self.inputs: batch_images,
            self.z: batch_z
        })[1]
    # Write summary.
    self.writer.add_summary(summary_str, counter)
