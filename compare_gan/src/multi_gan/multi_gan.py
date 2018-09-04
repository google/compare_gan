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

"""Multiple GANs that together model the data distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src import params
from compare_gan.src.gans import consts
from compare_gan.src.gans import ops
from compare_gan.src.gans.gans_with_penalty import AbstractGANWithPenalty

import numpy as np
import tensorflow as tf


class MultiGAN(AbstractGANWithPenalty):
  """A GAN that combines several copies of itself to generate data."""

  def __init__(self, dataset_content, parameters, runtime_info,
               model_name="MultiGAN"):
    super(MultiGAN, self).__init__(
        dataset_content=dataset_content,
        parameters=parameters,
        runtime_info=runtime_info,
        model_name=model_name)

    self.k = parameters["k"]   # Number of copies to sum over.
    self.aggregate = parameters["aggregate"]  # How to aggregate the output.

    # Relational params
    self.n_heads = parameters.get("n_heads", 0)
    self.n_blocks = parameters.get("n_blocks", 0)
    self.share_block_weights = parameters.get("share_block_weights", False)
    self.embedding_dim = parameters.get("embedding_dim", 100)

  def build_model(self, is_training=True):
    image_dims = [self.input_height, self.input_width, self.c_dim]
    batch_size = self.batch_size

    # Input images.
    self.inputs = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_images")

    # Noise vector.
    self.z = tf.placeholder(
        tf.float32, [batch_size, self.k, self.z_dim], name="z")

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

  def attention_block(self, entities, reuse, name="attention_block"):
    """Performs non-local pairwise relational computations.

    Args:
      entities: A tensor of shape (B, K, D) where K is the number of entities.
      reuse: Whether to reuse the weights.
      name: The name of the block.

    Returns:
      Updated entity representation (B, K, D)
    """
    # Estimate local dimensions to support background channel.
    k, z_dim = entities.get_shape().as_list()[1:3]

    r_entities = tf.reshape(entities, [self.batch_size * k, z_dim])

    with tf.variable_scope(name, reuse=reuse):
      queries = ops.layer_norm(tf.nn.relu(ops.linear(
          r_entities, self.embedding_dim, scope="q_fc")), reuse, "q_ln")
      queries = tf.reshape(
          queries, [self.batch_size, k, self.embedding_dim])

      keys = ops.layer_norm(tf.nn.relu(ops.linear(
          r_entities, self.embedding_dim, scope="k_fc")), reuse, "k_ln")
      keys = tf.reshape(keys, [self.batch_size, k, self.embedding_dim])

      values = ops.layer_norm(tf.nn.relu(ops.linear(
          r_entities, self.embedding_dim, scope="v_fc")), reuse, "v_ln")
      values = tf.reshape(values, [self.batch_size, k, self.embedding_dim])

      attention_weights = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))
      norm_attention_weights = tf.nn.softmax(
          attention_weights / tf.sqrt(
              tf.cast(self.embedding_dim, tf.float32)), axis=2)

      attention = tf.matmul(norm_attention_weights, values)
      r_attention = tf.reshape(attention, [
          self.batch_size * k, self.embedding_dim])

      # Project back to original space.
      u_entities = tf.nn.relu(ops.linear(r_attention, z_dim, "e_fc1"))
      u_entities = tf.nn.relu(ops.linear(u_entities, z_dim, "e_fc2"))
      u_entities = ops.layer_norm(u_entities + r_entities, reuse, "e_ln")

      return tf.reshape(u_entities, [self.batch_size, k, z_dim])

  def aggregate_heads(self, heads, reuse, name="aggregate_heads"):
    """Returns the aggregated heads."""
    # Estimate local dimensions to support background channel.
    k, z_dim = heads[0].get_shape().as_list()[1:3]

    with tf.variable_scope(name, reuse=reuse):
      heads = tf.concat(heads, axis=2)
      heads_r = tf.reshape(heads, [
          self.batch_size * k, self.n_heads * z_dim])
      heads_a = tf.nn.relu(ops.linear(
          tf.concat(heads_r, axis=2), z_dim, "a_fc1"))
      heads_a = ops.layer_norm(heads_a, reuse, "a_ln")
      heads_a = tf.reshape(
          heads_a, [self.batch_size, k, z_dim])

      return heads_a

  def update_z(self, z, reuse):
    """Returns an updated z by using MHA to compute relations among them."""

    # Update each latent representation based on others
    for block in range(self.n_blocks):
      share_params = reuse or (self.share_block_weights and block > 0)
      block_name = "generator_att_block"
      block_name += str(block) if not self.share_block_weights else ""

      z_s = []
      for head in range(self.n_heads):
        head_name = "%s-head%d" % (block_name, head)
        z_s.append(self.attention_block(z, share_params, name=head_name))

      # [B, K, Z_dim * n_heads] -> [B, K, Z_dim]
      if self.n_heads > 1:
        z = self.aggregate_heads(
            z_s, share_params, block_name + "-aggregate_heads")
      else:
        z = z_s[0]

    return z

  def aggregate_images(self, generated):
    """Returns the aggregated image from the generated images by each comp."""
    # Estimate local k / c_dim to support background / alpha channel.
    k = generated.get_shape().as_list()[1]
    c_dim = 3 if generated.get_shape().as_list()[-1] >= 3 else 1

    # Aggregate generator output.
    if self.aggregate == "sum_clip":
      aggregated = tf.reduce_sum(generated, axis=1)
      aggregated = tf.clip_by_value(aggregated, 0.0, 1.0)
    # Aggregate via alpha compositing, either using a generated alpha or an
    # implicit one obtained via thresholding.
    elif self.aggregate in ["alpha", "implicit_alpha"]:
      def AOverB(color_a_, alpha_a_, color_b_, alpha_b_):
        """Returns A OVER B operation using alpha compositing."""
        alpha_o = alpha_a_ + alpha_b_ * (1. - alpha_a_)
        color_o = (color_a_ * alpha_a_ + (
            color_b_ * alpha_b_ * (1. - alpha_a_))) / alpha_o

        return color_o, alpha_o

      # Start with last map to match stacking order when generating background.
      color_b = generated[:, -1, :, :, :c_dim]
      if self.aggregate == "implicit_alpha":
        alpha_shape = color_b.get_shape().as_list()[:-1] + [1]
        alpha_b = tf.ones(alpha_shape, dtype=tf.float32)
      else:
        alpha_b = generated[:, -1, :, :, -1:]

      for i in range(k-1, -1, -1):
        color_a = generated[:, i, :, :, :c_dim]
        if self.aggregate == "implicit_alpha":
          epsilon = 0.1
          mask = tf.reduce_max(color_a, axis=-1, keep_dims=True) > epsilon
          alpha_a = tf.where(
              mask, tf.ones_like(alpha_b), tf.zeros_like(alpha_b))
        else:
          alpha_a = generated[:, i, :, :, -1:]

        # Perform A OVER B (non-premultiplied version)
        color_b, alpha_b = AOverB(color_a, alpha_a, color_b, alpha_b)

      aggregated = color_b
    else:
      raise ValueError("Unknown aggregation method: %s" % self.aggregate)

    return aggregated

  def generate_images(self, z, is_training, reuse):
    """Returns K generated images (B, K, W, H, C) from z."""
    # Update each latent representation based on others
    z = self.update_z(z, reuse)

    # Pass latent representation through generator.
    out = []
    for i in range(self.k):
      use_copy = reuse or (i > 0)
      out_k = super(MultiGAN, self).generator(z[:, i], is_training, use_copy)
      out.append(tf.expand_dims(out_k, 1))

    return tf.concat(out, axis=1, name="generator_predictions")

  def generator(self, z, is_training, reuse=False):
    # Hack to add alpha channel to generator output if aggregate = alpha
    if self.aggregate == "alpha":
      self.c_dim += 1

    # Generate and aggregate images to obtain final image
    generated = self.generate_images(z, is_training, reuse)
    aggregated = self.aggregate_images(generated)

    # Hack to reset alpha channel after use
    if self.aggregate == "alpha":
      self.c_dim -= 1

    return aggregated

  def z_generator(self, batch_size, z_dim):
    z = []
    for _ in range(self.k):
      z_k = super(MultiGAN, self).z_generator(batch_size, z_dim)
      z.append(z_k[:, None, :])

    return np.concatenate(z, axis=1)  # (batch_size, k, z_dim)

  def z_tf_generator(self, batch_size, z_dim, name=None):
    z = []
    for _ in range(self.k):
      z_k = super(MultiGAN, self).z_tf_generator(batch_size, z_dim, name)
      z.append(tf.expand_dims(z_k, 1))

    return tf.concat(z, axis=1)


def MultiGANHyperParams(range_type, gan_type, penalty_type):
  """Return a default set of hyperparameters for MultiGAN."""
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

  # GAN Penalty
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
  param_ranges.AddRange("aggregate", "sum_clip", ["sum_clip"],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("n_heads", 1, [1, 2, 3],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("n_blocks", 1, [1, 2, 3],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("share_block_weights", False, [True, False],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("embedding_dim", 32, [32, 64, 128],
                        is_log_scale=False, is_discrete=True)

  return param_ranges.GetParams()
