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

"""Resnet generator and discriminator for CIFAR.

Based on Table 4 from "Spectral Normalization for Generative Adversarial
Networks", Miyato T. et al., 2018. [https://arxiv.org/pdf/1802.05957.pdf].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import arch_ops as ops
from compare_gan.architectures import resnet_ops

import gin
from six.moves import range
import tensorflow as tf


@gin.configurable
class Generator(resnet_ops.ResNetGenerator):
  """ResNet generator, 4 blocks, supporting 32x32 resolution."""

  def __init__(self,
               hierarchical_z=False,
               embed_z=False,
               embed_y=False,
               **kwargs):
    """Constructor for the ResNet Cifar generator.

    Args:
      hierarchical_z: Split z into chunks and only give one chunk to each.
        Each chunk will also be concatenated to y, the one hot encoded labels.
      embed_z: If True use a learnable embedding of z that is used instead.
        The embedding will have the length of z.
      embed_y: If True use a learnable embedding of y that is used instead.
        The embedding will have the length of z (not y!).
      **kwargs: additional arguments past on to ResNetGenerator.
    """
    super(Generator, self).__init__(**kwargs)
    self._hierarchical_z = hierarchical_z
    self._embed_z = embed_z
    self._embed_y = embed_y

  def apply(self, z, y, is_training):
    """Build the generator network for the given inputs.

    Args:
      z: `Tensor` of shape [batch_size, z_dim] with latent code.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, are we in train or eval model.

    Returns:
      A tensor of size [batch_size, 32, 32, colors] with values in [0, 1].
    """
    assert self._image_shape[0] == 32
    assert self._image_shape[1] == 32
    num_blocks = 3
    z_dim = z.shape[1].value

    if self._embed_z:
      z = ops.linear(z, z_dim, scope="embed_z", use_sn=self._spectral_norm)
    if self._embed_y:
      y = ops.linear(y, z_dim, scope="embed_y", use_sn=self._spectral_norm)
    y_per_block = num_blocks * [y]
    if self._hierarchical_z:
      z_per_block = tf.split(z, num_blocks + 1, axis=1)
      z0, z_per_block = z_per_block[0], z_per_block[1:]
      if y is not None:
        y_per_block = [tf.concat([zi, y], 1) for zi in z_per_block]
    else:
      z0 = z
      z_per_block = num_blocks * [z]

    output = ops.linear(z0, 4 * 4 * 256, scope="fc_noise",
                        use_sn=self._spectral_norm)
    output = tf.reshape(output, [-1, 4, 4, 256], name="fc_reshaped")
    for block_idx in range(3):
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=256,
          out_channels=256,
          scale="up")
      output = block(
          output,
          z=z_per_block[block_idx],
          y=y_per_block[block_idx],
          is_training=is_training)

    # Final processing of the output.
    output = self.batch_norm(
        output, z=z, y=y, is_training=is_training, name="final_norm")
    output = tf.nn.relu(output)
    output = ops.conv2d(output, output_dim=self._image_shape[2], k_h=3, k_w=3,
                        d_h=1, d_w=1, name="final_conv",
                        use_sn=self._spectral_norm,)
    return tf.nn.sigmoid(output)


@gin.configurable
class Discriminator(resnet_ops.ResNetDiscriminator):
  """ResNet discriminator, 4 blocks, supporting 32x32 with 1 or 3 colors."""

  def __init__(self, project_y=False, **kwargs):
    super(Discriminator, self).__init__(**kwargs)
    self._project_y = project_y

  def apply(self, x, y, is_training):
    """Apply the discriminator on a input.

    Args:
      x: `Tensor` of shape [batch_size, 32, 32, ?] with real or fake images.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: Boolean, whether the architecture should be constructed for
        training or inference.

    Returns:
      Tuple of 3 Tensors, the final prediction of the discriminator, the logits
      before the final output activation function and logits form the second
      last layer.
    """
    resnet_ops.validate_image_inputs(x)
    colors = x.shape[3].value
    if colors not in [1, 3]:
      raise ValueError("Number of color channels not supported: {}".format(
          colors))

    output = x
    for block_idx in range(4):
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=colors if block_idx == 0 else 128,
          out_channels=128,
          scale="down" if block_idx <= 1 else "none")
      output = block(output, z=None, y=y, is_training=is_training)

    # Final part - ReLU
    output = tf.nn.relu(output)

    h = tf.reduce_mean(output, axis=[1, 2])

    out_logit = ops.linear(h, 1, scope="disc_final_fc",
                           use_sn=self._spectral_norm)
    if self._project_y:
      if y is None:
        raise ValueError("You must provide class information y to project.")
      embedded_y = ops.linear(y, 128, use_bias=False,
                              scope="embedding_fc", use_sn=self._spectral_norm)
      out_logit += tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, h
