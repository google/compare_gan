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

"""A deep neural architecture with residual blocks and skip connections.

Based on Table 5 from "Spectral Normalization for Generative Adversarial
Networks", Miyato T. et al., 2018. [https://arxiv.org/pdf/1802.05957.pdf].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import arch_ops as ops
from compare_gan.architectures import resnet_ops

from six.moves import range
import tensorflow as tf


class Generator(resnet_ops.ResNetGenerator):
  """ResNet generator, 3 blocks, supporting 48x48 resolution."""

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
    ch = 64
    colors = self._image_shape[2]
    batch_size = z.get_shape().as_list()[0]
    magic = [(8, 4), (4, 2), (2, 1)]
    output = ops.linear(z, 6 * 6 * 512, scope="fc_noise")
    output = tf.reshape(output, [batch_size, 6, 6, 512], name="fc_reshaped")
    for block_idx in range(3):
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=ch * magic[block_idx][0],
          out_channels=ch * magic[block_idx][1],
          scale="up")
      output = block(output, z=z, y=y, is_training=is_training)
    output = self.batch_norm(
        output, z=z, y=y, is_training=is_training, scope="final_norm")
    output = tf.nn.relu(output)
    output = ops.conv2d(output, output_dim=colors, k_h=3, k_w=3, d_h=1, d_w=1,
                        name="final_conv")
    return tf.nn.sigmoid(output)


class Discriminator(resnet_ops.ResNetDiscriminator):
  """ResNet discriminator, 4 blocks, suports 48x48 resolution."""

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
    resnet_ops.validate_image_inputs(x, validate_power2=False)
    colors = x.shape[-1].value
    if colors not in [1, 3]:
      raise ValueError("Number of color channels unknown: %s" % colors)
    ch = 64
    block = self._resnet_block(
        name="B0", in_channels=colors, out_channels=ch, scale="down")
    output = block(x, z=None, y=y, is_training=is_training)
    magic = [(1, 2), (2, 4), (4, 8), (8, 16)]
    for block_idx in range(4):
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=ch * magic[block_idx][0],
          out_channels=ch * magic[block_idx][1],
          scale="down" if block_idx < 3 else "none")
      output = block(output, z=None, y=y, is_training=is_training)
    output = tf.nn.relu(output)
    pre_logits = tf.reduce_mean(output, axis=[1, 2])
    out_logit = ops.linear(pre_logits, 1, scope="disc_final_fc",
                           use_sn=self._spectral_norm)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, pre_logits
