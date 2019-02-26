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

"""A 30-block resnet.

It contains 6 "super-blocks" and each such block contains 5 residual blocks in
both the generator and discriminator. It supports the 128x128 resolution.
Details can be found in "Improved Training of Wasserstein GANs", Gulrajani I.
et al. 2017. The related code is available at
https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import arch_ops as ops
from compare_gan.architectures import resnet_ops

from six.moves import range
import tensorflow as tf


class Generator(resnet_ops.ResNetGenerator):
  """ResNet30 generator, 30 blocks, generates images of resolution 128x128.

  Trying to match the architecture defined in [1]. Difference is that there
  the final resolution is 64x64, while here we have 128x128.
  """

  def apply(self, z, y, is_training):
    """Build the generator network for the given inputs.

    Args:
      z: `Tensor` of shape [batch_size, z_dim] with latent code.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, are we in train or eval model.

    Returns:
      A tensor of size [batch_size] + self._image_shape with values in [0, 1].
    """
    z_shape = z.get_shape().as_list()
    if len(z_shape) != 2:
      raise ValueError("Expected shape [batch_size, z_dim], got %s." % z_shape)
    ch = 64
    colors = self._image_shape[2]
    # Map noise to the actual seed.
    output = ops.linear(z, 4 * 4 * 8 * ch, scope="fc_noise")
    # Reshape the seed to be a rank-4 Tensor.
    output = tf.reshape(output, [-1, 4, 4, 8 * ch], name="fc_reshaped")
    in_channels = 8 * ch
    out_channels = 4 * ch
    for superblock in range(6):
      for i in range(5):
        block = self._resnet_block(
            name="B_{}_{}".format(superblock, i),
            in_channels=in_channels,
            out_channels=in_channels,
            scale="none")
        output = block(output, z=z, y=y, is_training=is_training)
      # We want to upscale 5 times.
      if superblock < 5:
        block = self._resnet_block(
            name="B_{}_up".format(superblock),
            in_channels=in_channels,
            out_channels=out_channels,
            scale="up")
        output = block(output, z=z, y=y, is_training=is_training)
      in_channels /= 2
      out_channels /= 2

    output = ops.conv2d(
        output, output_dim=colors, k_h=3, k_w=3, d_h=1, d_w=1,
        name="final_conv")
    output = tf.nn.sigmoid(output)
    return output


class Discriminator(resnet_ops.ResNetDiscriminator):
  """ResNet discriminator, 30 blocks, 128x128x3 and 128x128x1 resolution."""

  def apply(self, x, y, is_training):
    """Apply the discriminator on a input.

    Args:
      x: `Tensor` of shape [batch_size, ?, ?, ?] with real or fake images.
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
    colors = x.get_shape().as_list()[-1]
    assert colors in [1, 3]
    ch = 64
    output = ops.conv2d(
        x, output_dim=ch // 4, k_h=3, k_w=3, d_h=1, d_w=1,
        name="color_conv")
    in_channels = ch // 4
    out_channels = ch // 2
    for superblock in range(6):
      for i in range(5):
        block = self._resnet_block(
            name="B_{}_{}".format(superblock, i),
            in_channels=in_channels,
            out_channels=in_channels,
            scale="none")
        output = block(output, z=None, y=y, is_training=is_training)
      # We want to downscale 5 times.
      if superblock < 5:
        block = self._resnet_block(
            name="B_{}_up".format(superblock),
            in_channels=in_channels,
            out_channels=out_channels,
            scale="down")
        output = block(output, z=None, y=y, is_training=is_training)
      in_channels *= 2
      out_channels *= 2

    # Final part
    output = tf.reshape(output, [-1, 4 * 4 * 8 * ch])
    out_logit = ops.linear(output, 1, scope="disc_final_fc",
                           use_sn=self._spectral_norm)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, output
