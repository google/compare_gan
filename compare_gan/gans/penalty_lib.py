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

"""Implementation of popular GAN penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan import utils
from compare_gan.gans import ops
import gin
import tensorflow as tf


@gin.configurable
def no_penalty():
  return tf.constant(0.0)


@gin.configurable(whitelist=[])
def dragan_penalty(discriminator, x, y, is_training):
  """Returns the DRAGAN gradient penalty.

  Args:
    discriminator: Instance of `AbstractDiscriminator`.
    x: Samples from the true distribution, shape [bs, h, w, channels].
    y: Encoded class embedding for the samples. None for unsupervised models.
    is_training: boolean, are we in train or eval model.

  Returns:
    A tensor with the computed penalty.
  """
  with tf.name_scope("dragan_penalty"):
    _, var = tf.nn.moments(x, axes=list(range(len(x.get_shape()))))
    std = tf.sqrt(var)
    x_noisy = x + std * (ops.random_uniform(x.shape) - 0.5)
    x_noisy = tf.clip_by_value(x_noisy, 0.0, 1.0)
    logits = discriminator(x_noisy, y=y, is_training=is_training, reuse=True)[1]
    gradients = tf.gradients(logits, [x_noisy])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
    return gradient_penalty


@gin.configurable(whitelist=[])
def wgangp_penalty(discriminator, x, x_fake, y, is_training):
  """Returns the WGAN gradient penalty.

  Args:
    discriminator: Instance of `AbstractDiscriminator`.
    x: samples from the true distribution, shape [bs, h, w, channels].
    x_fake: samples from the fake distribution, shape [bs, h, w, channels].
    y: Encoded class embedding for the samples. None for unsupervised models.
    is_training: boolean, are we in train or eval model.

  Returns:
    A tensor with the computed penalty.
  """
  with tf.name_scope("wgangp_penalty"):
    alpha = ops.random_uniform(shape=[x.shape[0].value, 1, 1, 1], name="alpha")
    interpolates = x + alpha * (x_fake - x)
    logits = discriminator(
        interpolates, y=y, is_training=is_training, reuse=True)[1]
    gradients = tf.gradients(logits, [interpolates])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
    return gradient_penalty


@gin.configurable(whitelist=[])
def l2_penalty(discriminator):
  """Returns the L2 penalty for each matrix/vector excluding biases.

  Assumes a specific tensor naming followed throughout the compare_gan library.
  We penalize all fully connected, conv2d, and deconv2d layers.

  Args:
    discriminator: Instance of `AbstractDiscriminator`.

  Returns:
     A tensor with the computed penalty.
  """
  with tf.name_scope("l2_penalty"):
    d_weights = [v for v in discriminator.trainable_variables
                 if v.name.endswith("/kernel:0")]
    return tf.reduce_mean(
        [tf.nn.l2_loss(i) for i in d_weights], name="l2_penalty")


@gin.configurable("penalty", whitelist=["fn"])
def get_penalty_loss(fn=no_penalty, **kwargs):
  """Returns the penalty loss."""
  return utils.call_with_accepted_args(fn, **kwargs)
