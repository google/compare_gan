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

It contains 5 residual blocks in both the generator and discriminator and
supports 128x128 resolution. Details can be found in "Improved Training
of Wasserstein GANs", Gulrajani I. et al. 2017. The related code is available at
https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import arch_ops as ops
from compare_gan.architectures import resnet_ops

import numpy as np
from six.moves import range
import tensorflow as tf


class Generator(resnet_ops.ResNetGenerator):
  """ResNet generator consisting of 5 blocks, outputs 128x128x3 resolution."""

  def __init__(self, ch=64, channels=(8, 8, 4, 4, 2, 1), **kwargs):
    super(Generator, self).__init__(**kwargs)
    self._ch = ch
    self._channels = channels

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
    # Each block upscales by a factor of 2.
    seed_size = 4
    image_size = self._image_shape[0]

    # Map noise to the actual seed.
    net = ops.linear(
        z,
        self._ch * self._channels[0] * seed_size * seed_size,
        scope="fc_noise")
    # Reshape the seed to be a rank-4 Tensor.
    net = tf.reshape(
        net,
        [-1, seed_size, seed_size, self._ch * self._channels[0]],
        name="fc_reshaped")

    up_layers = np.log2(float(image_size) / seed_size)
    if not up_layers.is_integer():
      raise ValueError("log2({}/{}) must be an integer.".format(
          image_size, seed_size))
    if up_layers < 0 or up_layers > 5:
      raise ValueError("Invalid image_size {}.".format(image_size))
    up_layers = int(up_layers)

    for block_idx in range(5):
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=self._ch * self._channels[block_idx],
          out_channels=self._ch * self._channels[block_idx + 1],
          scale="up" if block_idx < up_layers else "none")
      net = block(net, z=z, y=y, is_training=is_training)

    net = self.batch_norm(
        net, z=z, y=y, is_training=is_training, name="final_norm")
    net = tf.nn.relu(net)
    net = ops.conv2d(net, output_dim=self._image_shape[2],
                     k_h=3, k_w=3, d_h=1, d_w=1, name="final_conv")
    net = tf.nn.sigmoid(net)
    return net


class Discriminator(resnet_ops.ResNetDiscriminator):
  """ResNet5 discriminator, 5 blocks, supporting 128x128x3 and 128x128x1."""

  def __init__(self, ch=64, channels=(1, 2, 4, 4, 8, 8), **kwargs):
    super(Discriminator, self).__init__(**kwargs)
    self._ch = ch
    self._channels = channels

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
    colors = x.shape[3].value
    if colors not in [1, 3]:
      raise ValueError("Number of color channels not supported: {}".format(
          colors))

    block = self._resnet_block(
        name="B0",
        in_channels=colors,
        out_channels=self._ch,
        scale="down")
    output = block(x, z=None, y=y, is_training=is_training)

    for block_idx in range(5):
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=self._ch * self._channels[block_idx],
          out_channels=self._ch * self._channels[block_idx + 1],
          scale="down")
      output = block(output, z=None, y=y, is_training=is_training)

    output = tf.nn.relu(output)
    pre_logits = tf.reduce_mean(output, axis=[1, 2])
    out_logit = ops.linear(pre_logits, 1, scope="disc_final_fc",
                           use_sn=self._spectral_norm)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, pre_logits
