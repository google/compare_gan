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

"""Implementation of Self-Supervised GAN with auxiliary rotation loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
from absl import logging
from compare_gan.architectures import resnet5
from compare_gan.architectures import resnet5_biggan
from compare_gan.architectures import resnet_cifar
from compare_gan.architectures import sndcgan
from compare_gan.architectures.arch_ops import linear
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans import modular_gan
from compare_gan.gans import penalty_lib
from compare_gan.gans import utils

import gin
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
NUM_ROTATIONS = 4


@gin.configurable(blacklist=["kwargs"])
class SSGAN(modular_gan.ModularGAN):
  """Self-Supervised GAN.

  http://arxiv.org/abs/1811.11212
  """

  def __init__(self,
               self_supervision="rotation_gan",
               rotated_batch_size=gin.REQUIRED,
               weight_rotation_loss_d=1.0,
               weight_rotation_loss_g=0.2,
               **kwargs):
    """Creates a new Self-Supervised GAN.

    Args:
      self_supervision: One of [rotation_gan, rotation_only, None]. When it is
        rotation_only, no GAN loss is used, degenerates to a pure rotation
        model.
      rotated_batch_size: The total number images per batch for the rotation
        loss. This must be a multiple of (4 * #CORES) since we consider 4
        rotations of each images on each TPU core. For GPU training #CORES is 1.
      weight_rotation_loss_d: Weight for the rotation loss for the discriminator
        on real images.
      weight_rotation_loss_g: Weight for the rotation loss for the generator
        on fake images.
      **kwargs: Additional arguments passed to `ModularGAN` constructor.
    """
    super(SSGAN, self).__init__(**kwargs)

    self._self_supervision = self_supervision
    self._rotated_batch_size = rotated_batch_size
    self._weight_rotation_loss_d = weight_rotation_loss_d
    self._weight_rotation_loss_g = weight_rotation_loss_g

  def discriminator(self, x, y, is_training, reuse=False, rotation_head=False):
    """Discriminator network with augmented auxiliary predictions.

    Args:
      x: an input image tensor.
      y: Tensor with label indices.
      is_training: boolean, whether or not it is a training call.
      reuse: boolean, whether or not to reuse the variables.
      rotation_head: If True add a rotation head on top of the discriminator
        logits.

    Returns:
      real_probs: the [0, 1] probability tensor of x being real images.
      real_scores: the unbounded score tensor of x being real images.
      rotation_scores: the categorical probablity of x being rotated in one of
        the four directions.
    """
    if not rotation_head:
      return super(SSGAN, self).discriminator(
          x, y=y, is_training=is_training, reuse=reuse)

    real_probs, real_scores, final = super(SSGAN, self).discriminator(
        x, y=y, is_training=is_training, reuse=reuse)

    # Hack to get whether to use spectral norm for the rotation head below.
    # Spectral norm is configured on the architecture (AbstractGenerator or
    # AbstrtactDiscriminator). The layer below is be part of the architecture.

    discriminator = {
        c.RESNET5_ARCH: resnet5.Discriminator,
        c.RESNET5_BIGGAN_ARCH: resnet5_biggan.Discriminator,
        c.RESNET_CIFAR: resnet_cifar.Discriminator,
        c.SNDCGAN_ARCH: sndcgan.Discriminator,
    }[self._architecture]()
    use_sn = discriminator._spectral_norm  # pylint: disable=protected-access

    with tf.variable_scope("discriminator_rotation", reuse=reuse):
      rotation_scores = linear(tf.reshape(final, (tf.shape(x)[0], -1)),
                               NUM_ROTATIONS,
                               scope="score_classify",
                               use_sn=use_sn)
    return real_probs, real_scores, rotation_scores

  def create_loss(self, features, labels, params, is_training=True,
                  reuse=False):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss and self.g_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      labels: Tensor will labels. These are class indices. Use
          self._get_one_hot_labels(labels) to get a one hot encoded tensor.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).
      reuse: Bool, whether to reuse existing variables for the models.
          This is only used for unrolling discriminator iterations when training
          on TPU.

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    images = features["images"]  # Input images.
    if self.conditional:
      y = self._get_one_hot_labels(labels)
      sampled_y = self._get_one_hot_labels(features["sampled_labels"])
    else:
      y = None
      sampled_y = None
      all_y = None

    if self._experimental_joint_gen_for_disc:
      assert "generated" in features
      generated = features["generated"]
    else:
      logging.warning("Computing fake images for every sub step separately.")
      z = features["z"]  # Noise vector.
      generated = self.generator(
          z, y=sampled_y, is_training=is_training, reuse=reuse)

    # Batch size per core.
    bs = params["batch_size"] // self.num_sub_steps
    num_replicas = params["context"].num_replicas if "context" in params else 1
    assert self._rotated_batch_size % num_replicas == 0
    # Rotated batch size per core.
    rotated_bs = self._rotated_batch_size // num_replicas
    assert rotated_bs % 4 == 0
    # Number of images to rotate. Each images gets rotated 3 times.
    num_rotated_examples = rotated_bs // 4
    logging.info("num_replicas=%s, bs=%s, rotated_bs=%s, "
                 "num_rotated_examples=%s, params=%s",
                 num_replicas, bs, rotated_bs, num_rotated_examples, params)

    # Augment the images with rotation.
    if "rotation" in self._self_supervision:
      # Put all rotation angles in a single batch, the first batch_size are
      # the original up-right images, followed by rotated_batch_size * 3
      # rotated images with 3 different angles.
      assert num_rotated_examples <= bs, (num_rotated_examples, bs)
      images_rotated = utils.rotate_images(
          images[-num_rotated_examples:], rot90_scalars=(1, 2, 3))
      generated_rotated = utils.rotate_images(
          generated[-num_rotated_examples:], rot90_scalars=(1, 2, 3))
      # Labels for rotation loss (unrotated and 3 rotated versions). For
      # NUM_ROTATIONS=4 and num_rotated_examples=2 this is:
      # [0, 0, 1, 1, 2, 2, 3, 3]
      rotate_labels = tf.constant(
          np.repeat(np.arange(NUM_ROTATIONS, dtype=np.int32),
                    num_rotated_examples))
      rotate_labels_onehot = tf.one_hot(rotate_labels, NUM_ROTATIONS)
      all_images = tf.concat([images, images_rotated,
                              generated, generated_rotated], 0)
      if self.conditional:
        y_rotated = tf.tile(y[-num_rotated_examples:], [3, 1])
        sampled_y_rotated = tf.tile(y[-num_rotated_examples:], [3, 1])
        all_y = tf.concat([y, y_rotated, sampled_y, sampled_y_rotated], 0)
    else:
      all_images = tf.concat([images, generated], 0)
      if self.conditional:
        all_y = tf.concat([y, sampled_y], axis=0)

    # Compute discriminator output for real and fake images in one batch.
    d_all, d_all_logits, c_all_logits = self.discriminator(
        all_images, y=all_y, is_training=is_training, reuse=reuse,
        rotation_head=True)
    d_real, d_fake = tf.split(d_all, 2)
    d_real_logits, d_fake_logits = tf.split(d_all_logits, 2)
    c_real_logits, c_fake_logits = tf.split(c_all_logits, 2)

    # Separate the true/fake scores from whole rotation batch.
    d_real_logits = d_real_logits[:bs]
    d_fake_logits = d_fake_logits[:bs]
    d_real = d_real[:bs]
    d_fake = d_fake[:bs]

    self.d_loss, _, _, self.g_loss = loss_lib.get_losses(
        d_real=d_real, d_fake=d_fake, d_real_logits=d_real_logits,
        d_fake_logits=d_fake_logits)

    discriminator = functools.partial(self.discriminator, y=y)
    penalty_loss = penalty_lib.get_penalty_loss(
        x=images, x_fake=generated, is_training=is_training,
        discriminator=discriminator, architecture=self._architecture)
    self.d_loss += self._lambda * penalty_loss

    # Add rotation augmented loss.
    if "rotation" in self._self_supervision:
      # We take an even pieces for all rotation angles
      assert len(c_real_logits.shape.as_list()) == 2, c_real_logits.shape
      assert len(c_fake_logits.shape.as_list()) == 2, c_fake_logits.shape
      c_real_logits = c_real_logits[- rotated_bs:]
      c_fake_logits = c_fake_logits[- rotated_bs:]
      preds_onreal = tf.cast(tf.argmax(c_real_logits, -1), rotate_labels.dtype)
      accuracy = tf.reduce_mean(
          tf.cast(tf.equal(rotate_labels, preds_onreal), tf.float32))
      c_real_probs = tf.nn.softmax(c_real_logits)
      c_fake_probs = tf.nn.softmax(c_fake_logits)
      c_real_loss = - tf.reduce_mean(
          tf.reduce_sum(rotate_labels_onehot * tf.log(c_real_probs + 1e-10), 1))
      c_fake_loss = - tf.reduce_mean(
          tf.reduce_sum(rotate_labels_onehot * tf.log(c_fake_probs + 1e-10), 1))
      if self._self_supervision == "rotation_only":
        self.d_loss *= 0.0
        self.g_loss *= 0.0
      self.d_loss += c_real_loss * self._weight_rotation_loss_d
      self.g_loss += c_fake_loss * self._weight_rotation_loss_g
    else:
      c_real_loss = 0.0
      c_fake_loss = 0.0
      accuracy = tf.zeros([])

    self._tpu_summary.scalar("loss/c_real_loss", c_real_loss)
    self._tpu_summary.scalar("loss/c_fake_loss", c_fake_loss)
    self._tpu_summary.scalar("accuracy/d_rotation", accuracy)
    self._tpu_summary.scalar("loss/d", self.d_loss)
    self._tpu_summary.scalar("loss/g", self.g_loss)
    self._tpu_summary.scalar("loss/penalty", penalty_loss)

