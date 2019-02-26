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

"""Implementation of popular GAN losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan import utils
import gin
import tensorflow as tf


def check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits):
  """Checks the shapes and ranks of logits and prediction tensors.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].

  Raises:
    ValueError: if the ranks or shapes are mismatched.
  """
  def _check_pair(a, b):
    if a != b:
      raise ValueError("Shape mismatch: %s vs %s." % (a, b))
    if len(a) != 2 or len(b) != 2:
      raise ValueError("Rank: expected 2, got %s and %s" % (len(a), len(b)))

  if (d_real is not None) and (d_fake is not None):
    _check_pair(d_real.shape.as_list(), d_fake.shape.as_list())
  if (d_real_logits is not None) and (d_fake_logits is not None):
    _check_pair(d_real_logits.shape.as_list(), d_fake_logits.shape.as_list())
  if (d_real is not None) and (d_real_logits is not None):
    _check_pair(d_real.shape.as_list(), d_real_logits.shape.as_list())


@gin.configurable(whitelist=[])
def non_saturating(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Non-saturating loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("non_saturating_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_logits, labels=tf.ones_like(d_real_logits),
        name="cross_entropy_d_real"))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits),
        name="cross_entropy_d_fake"))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.ones_like(d_fake_logits),
        name="cross_entropy_g"))
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def wasserstein(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Wasserstein loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("wasserstein_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = -tf.reduce_mean(d_real_logits)
    d_loss_fake = tf.reduce_mean(d_fake_logits)
    d_loss = d_loss_real + d_loss_fake
    g_loss = -d_loss_fake
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def least_squares(d_real, d_fake, d_real_logits=None, d_fake_logits=None):
  """Returns the discriminator and generator loss for the least-squares loss.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: ignored.
    d_fake_logits: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("least_square_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.square(d_real - 1.0))
    d_loss_fake = tf.reduce_mean(tf.square(d_fake))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    g_loss = 0.5 * tf.reduce_mean(tf.square(d_fake - 1.0))
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def hinge(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for the hinge loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("hinge_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
    d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
    d_loss = d_loss_real + d_loss_fake
    g_loss = - tf.reduce_mean(d_fake_logits)
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable("loss", whitelist=["fn"])
def get_losses(fn=non_saturating, **kwargs):
  """Returns the losses for the discriminator and generator."""
  return utils.call_with_accepted_args(fn, **kwargs)
