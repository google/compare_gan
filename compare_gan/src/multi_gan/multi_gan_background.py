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

"""Multiple GANs that together model the data distribution, including BG."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src import params
from compare_gan.src.gans import consts
from compare_gan.src.multi_gan.multi_gan import MultiGAN

import numpy as np
import tensorflow as tf


class MultiGANBackground(MultiGAN):
  """A GAN consisting of a background generator and multiple copies of
     object generators."""

  def __init__(self, dataset_content, parameters, runtime_info):
    super(MultiGANBackground, self).__init__(
        dataset_content=dataset_content,
        parameters=parameters,
        runtime_info=runtime_info,
        model_name="MultiGANBackground")

    self.background_interaction = parameters["background_interaction"]

  def build_model(self, is_training=True):
    image_dims = [self.input_height, self.input_width, self.c_dim]
    batch_size = self.batch_size

    # Input images.
    self.inputs = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_images")

    # Noise vector.
    self.z = tf.placeholder(
        tf.float32, [batch_size, self.k + 1, self.z_dim], name="z")

    # Discriminator output for real images.
    d_real, d_real_logits, _ = self.discriminator(
        self.inputs, is_training=is_training, reuse=False)

    # Discriminator output for fake images.
    generated = self.generator(self.z, is_training=is_training, reuse=False)
    d_fake, d_fake_logits, _ = self.discriminator(
        generated, is_training=is_training, reuse=True)

    self.discriminator_output = self.discriminator(
        self.inputs, is_training=is_training, reuse=True)[0]

    # Define the loss functions (NS-GAN)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_real_logits, labels=tf.ones_like(d_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_logits, labels=tf.zeros_like(d_fake)))
    self.d_loss = d_loss_real + d_loss_fake
    self.g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_logits, labels=tf.ones_like(d_fake)))

    # Define the penalty.
    if self.penalty_type == consts.NO_PENALTY:
      self.penalty_loss = 0.0
    elif self.penalty_type == consts.DRAGAN_PENALTY:
      self.penalty_loss = self.dragan_penalty(self.inputs, self.discriminator,
                                              is_training)
      self.d_loss += self.lambd * self.penalty_loss
    elif self.penalty_type == consts.WGANGP_PENALTY:
      self.penalty_loss = self.wgangp_penalty(self.inputs, generated,
                                              self.discriminator, is_training)
      self.d_loss += self.lambd * self.penalty_loss
    elif self.penalty_type == consts.L2_PENALTY:
      self.penalty_loss = self.l2_penalty()
      self.d_loss += self.lambd * self.penalty_loss
    else:
      raise NotImplementedError(
          "The penalty %s was not implemented." % self.penalty_type)

    # Divide trainable variables into a group for D and group for G.
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    g_vars = [var for var in t_vars if "generator" in var.name]
    self.check_variables(t_vars, d_vars, g_vars)

    # Define optimization ops.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      self.d_optim = self.get_optimizer("d_").minimize(
          self.d_loss, var_list=d_vars)
      self.g_optim = self.get_optimizer("g_").minimize(
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

  def generate_images(self, z, is_training, reuse):
    """Returns generated object / backgrounds images and z's."""
    # Let z_b participate in relational part
    if self.background_interaction:
      z = self.update_z(z, reuse)

      # Isolate z used for background ALWAYS IN LAST
      z_s = tf.unstack(z, axis=1)
      z_o = tf.stack(z_s[:-1], axis=1)
      z_b = z_s[-1]

      # Pass latent representation through generator (copy from MG).
      out = []
      for i in range(self.k):
        use_copy = reuse or (i > 0)
        out_k = super(MultiGAN, self).generator(  # pylint: disable=bad-super-call
            z_o[:, i], is_training, use_copy)
        out.append(tf.expand_dims(out_k, 1))

      generated_o = tf.concat(out, axis=1, name="generator_predictions")
    # Only let z_o participate in relational part
    else:
      # Isolate z used for background ALWAYS IN LAST
      z_s = tf.unstack(z, axis=1)
      z_o = tf.stack(z_s[:-1], axis=1)
      z_b = z_s[-1]

      # Generated object images and background image
      generated_o = super(MultiGANBackground, self).generate_images(
          z_o, is_training, reuse)

    with tf.variable_scope("background_generator", reuse=reuse):
      generated_b = super(MultiGAN, self).generator(z_b, is_training, reuse)  # pylint: disable=bad-super-call

    return generated_o, generated_b, z_o, z_b

  def generator(self, z, is_training, reuse=False):
    # Hack to add alpha channel to generator output if aggregate = alpha
    if self.aggregate == "alpha":
      self.c_dim += 1

    # # Generated object images and background image.
    generated_o, generated_b, z_o, z_b = self.generate_images(
        z, is_training, reuse)

    # Hack to reset alpha channel after use
    # Add opaque alpha channel in case of alpha compositing to background
    if self.aggregate == "alpha":
      self.c_dim -= 1
      generated_b = tf.concat(
          (generated_b[..., :-1], tf.ones_like(generated_b[..., -1:])), axis=-1)

    # Aggregate and generated outputs (order matters for alpha / fixed_perm)
    z = tf.concat([z_o, tf.expand_dims(z_b, axis=1)], axis=1)
    generated = tf.concat(
        [generated_o, tf.expand_dims(generated_b, axis=1)], axis=1)
    aggregated = self.aggregate_images(generated)

    return aggregated

  def z_generator(self, batch_size, z_dim):
    z_o = super(MultiGANBackground, self).z_generator(batch_size, z_dim)
    z_b = np.random.uniform(-1, 1, size=(batch_size, 1, z_dim))

    return np.concatenate([z_b, z_o], axis=1)  # (batch_size, k + 1, z_dim)

  def z_tf_generator(self, batch_size, z_dim, name=None):
    z_o = super(MultiGANBackground, self).z_tf_generator(
        batch_size, z_dim, name)
    z_b = tf.random_uniform(
        (batch_size, 1, z_dim), minval=-1.0, maxval=1.0, name=name)

    return tf.concat([z_b, z_o], axis=1)  # (batch_size, k + 1, z_dim)


def MultiGANBackgroundHyperParams(range_type, gan_type, penalty_type):
  """Return a default set of hyperparameters for MultiGANBackground."""
  del gan_type

  param_ranges = params.GetDefaultRange(range_type)
  param_ranges.AddRange("penalty_type", consts.NO_PENALTY,
                        [consts.NO_PENALTY, consts.WGANGP_PENALTY],
                        is_log_scale=False, is_discrete=True)
  if penalty_type and penalty_type == consts.L2_PENALTY:
    param_ranges.AddRange("lambda", 0.01, [-4.0, 1.0],
                          is_log_scale=True, is_discrete=False)
  else:
    param_ranges.AddRange("lambda", 10.0, [-1.0, 2.0],
                          is_log_scale=True, is_discrete=False)
  param_ranges.AddRange("beta2", 0.999, [0, 1],
                        is_log_scale=False, is_discrete=False)

  # MultiGAN
  param_ranges.UpdateDefaults(
      {"beta1": 0.0, "learning_rate": 0.00005, "disc_iters": 1,
       "discriminator_normalization": consts.BATCH_NORM})
  param_ranges.AddRange("penalty_type", consts.NO_PENALTY,
                        [consts.NO_PENALTY, consts.WGANGP_PENALTY],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("discriminator_normalization", consts.NO_NORMALIZATION,
                        [consts.NO_NORMALIZATION, consts.BATCH_NORM,
                         consts.SPECTRAL_NORM],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("z_dim", 64, [64],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("k", 3, [1, 2, 3, 4, 5],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("aggregate", "sum_clip", ["sum", "sum_clip", "mean"],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("n_heads", 1, [1, 2, 3],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("n_blocks", 1, [1, 2, 3],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("share_block_weights", False, [True, False],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("embedding_dim", 32, [32, 64, 128],
                        is_log_scale=False, is_discrete=True)

  # MultiGANBackground
  param_ranges.AddRange("background_interaction", False, [True, False],
                        is_log_scale=False, is_discrete=True)

  return param_ranges.GetParams()
