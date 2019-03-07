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

"""Re-implementation of BigGAN-Deep architecture.

Disclaimer: We note that this is our best-effort re-implementation and stress
that even minor implementation differences may lead to large differences in
trained models due to sensitivity of GANs to optimization hyperparameters and
details of neural architectures. That being said, this code suffices to
reproduce the reported FID on ImageNet 128x128.

Based on "Large Scale GAN Training for High Fidelity Natural Image Synthesys",
Brock A. et al., 2018 [https://arxiv.org/abs/1809.11096].

Supported resolutions: 32, 64, 128, 256, 512.

Differences to original BigGAN:
- The self-attention block is always applied at a resolution of 64x64.
- Each batch norm gets the concatenation of the z and the class embedding. z is
  not chunked.
- The channel width multiplier defaults to 128 instead of 96.
- Double amount of ResNet blocks:
- Residual blocks have bottlenecks:
  - 1x1 convolution reduces number of channels before the 3x3 convolutions.
  - After the 3x3 convolutions a 1x1 creates the desired number of out channels.
- Skip connections in residual connections preserve identity (no 1x1 conv).
  - In G, to reduce the number of channels, drop additional chhannels.
  - In D add more channels by doing a 1x1 convolution.
- Mean instead of sum pooling in D after the final ReLU.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging

from compare_gan.architectures import abstract_arch
from compare_gan.architectures import arch_ops as ops
from compare_gan.architectures import resnet_ops

import gin
from six.moves import range
import tensorflow as tf


@gin.configurable
class BigGanDeepResNetBlock(object):
  """ResNet block with bottleneck and identity preserving skip connections."""

  def __init__(self,
               name,
               in_channels,
               out_channels,
               scale,
               spectral_norm=False,
               batch_norm=None):
    """Constructs a new ResNet block with bottleneck.

    Args:
      name: Scope name for the resent block.
      in_channels: Integer, the input channel size.
      out_channels: Integer, the output channel size.
      scale: Whether or not to scale up or down, choose from "up", "down" or
        "none".
      spectral_norm: Use spectral normalization for all weights.
      batch_norm: Function for batch normalization.
    """
    assert scale in ["up", "down", "none"]
    self._name = name
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._scale = scale
    self._spectral_norm = spectral_norm
    self.batch_norm = batch_norm

  def __call__(self, inputs, z, y, is_training):
    return self.apply(inputs=inputs, z=z, y=y, is_training=is_training)

  def _shortcut(self, inputs):
    """Constructs a skip connection from inputs."""
    with tf.variable_scope("shortcut", values=[inputs]):
      shortcut = inputs
      num_channels = inputs.shape[-1].value
      if num_channels > self._out_channels:
        assert self._scale == "up"
        # Drop redundant channels.
        logging.info("[Shortcut] Dropping %d channels in shortcut.",
                     num_channels - self._out_channels)
        shortcut = shortcut[:, :, :, :self._out_channels]
      if self._scale == "up":
        shortcut = resnet_ops.unpool(shortcut)
      if self._scale == "down":
        shortcut = tf.nn.pool(shortcut, [2, 2], "AVG", "SAME",
                              strides=[2, 2], name="pool")
      if num_channels < self._out_channels:
        assert self._scale == "down"
        # Increase number of channels if necessary.
        num_missing = self._out_channels - num_channels
        logging.info("[Shortcut] Adding %d channels in shortcut.", num_missing)
        added = ops.conv1x1(shortcut, num_missing, name="add_channels",
                            use_sn=self._spectral_norm)
        shortcut = tf.concat([shortcut, added], axis=-1)
      return shortcut

  def apply(self, inputs, z, y, is_training):
    """"ResNet block containing possible down/up sampling, shared for G / D.

    Args:
      inputs: a 3d input tensor of feature map.
      z: the latent vector for potential self-modulation. Can be None if use_sbn
        is set to False.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, whether or notthis is called during the training.

    Returns:
      output: a 3d output tensor of feature map.
    """
    if inputs.shape[-1].value != self._in_channels:
      raise ValueError(
          "Unexpected number of input channels (expected {}, got {}).".format(
              self._in_channels, inputs.shape[-1].value))

    bottleneck_channels = max(self._in_channels, self._out_channels) // 4
    bn = functools.partial(self.batch_norm, z=z, y=y, is_training=is_training)
    conv1x1 = functools.partial(ops.conv1x1, use_sn=self._spectral_norm)
    conv3x3 = functools.partial(ops.conv2d, k_h=3, k_w=3, d_h=1, d_w=1,
                                use_sn=self._spectral_norm)

    with tf.variable_scope(self._name, values=[inputs]):
      outputs = inputs

      with tf.variable_scope("conv1", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        outputs = conv1x1(outputs, bottleneck_channels, name="1x1_conv")

      with tf.variable_scope("conv2", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        if self._scale == "up":
          outputs = resnet_ops.unpool(outputs)
        outputs = conv3x3(outputs, bottleneck_channels, name="3x3_conv")

      with tf.variable_scope("conv3", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        outputs = conv3x3(outputs, bottleneck_channels, name="3x3_conv")

      with tf.variable_scope("conv4", values=[outputs]):
        outputs = bn(outputs, name="bn")
        outputs = tf.nn.relu(outputs)
        if self._scale == "down":
          outputs = tf.nn.pool(outputs, [2, 2], "AVG", "SAME", strides=[2, 2],
                               name="avg_pool")
        outputs = conv1x1(outputs, self._out_channels, name="1x1_conv")

      # Add skip-connection.
      outputs += self._shortcut(inputs)

      logging.info("[Block] %s (z=%s, y=%s) -> %s", inputs.shape,
                   None if z is None else z.shape,
                   None if y is None else y.shape, outputs.shape)
      return outputs


@gin.configurable
class Generator(abstract_arch.AbstractGenerator):
  """ResNet-based generator supporting resolutions 32, 64, 128, 256, 512."""

  def __init__(self,
               ch=128,
               embed_y=True,
               embed_y_dim=128,
               experimental_fast_conv_to_rgb=False,
               **kwargs):
    """Constructor for BigGAN generator.

    Args:
      ch: Channel multiplier.
      embed_y: If True use a learnable embedding of y that is used instead.
      embed_y_dim: Size of the embedding of y.
      experimental_fast_conv_to_rgb: If True optimize the last convolution to
        sacrifize memory for better speed.
      **kwargs: additional arguments past on to ResNetGenerator.
    """
    super(Generator, self).__init__(**kwargs)
    self._ch = ch
    self._embed_y = embed_y
    self._embed_y_dim = embed_y_dim
    self._experimental_fast_conv_to_rgb = experimental_fast_conv_to_rgb

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return BigGanDeepResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)

  def _get_in_out_channels(self):
    # See Table 7-9.
    resolution = self._image_shape[0]
    if resolution == 512:
      channel_multipliers = 4 * [16] + 4 * [8] + [4, 4, 2, 2, 1, 1, 1]
    elif resolution == 256:
      channel_multipliers = 4 * [16] + 4 * [8] + [4, 4, 2, 2, 1]
    elif resolution == 128:
      channel_multipliers = 4 * [16] + 2 * [8] + [4, 4, 2, 2, 1]
    elif resolution == 64:
      channel_multipliers = 4 * [16] + 2 * [8] + [4, 4, 2]
    elif resolution == 32:
      channel_multipliers = 8 * [4]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self._ch * c for c in channel_multipliers[:-1]]
    out_channels = [self._ch * c for c in channel_multipliers[1:]]
    return in_channels, out_channels

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
    shape_or_none = lambda t: None if t is None else t.shape
    logging.info("[Generator] inputs are z=%s, y=%s", z.shape, shape_or_none(y))
    seed_size = 4

    if self._embed_y:
      y = ops.linear(y, self._embed_y_dim, scope="embed_y", use_sn=False,
                     use_bias=False)
    if y is not None:
      y = tf.concat([z, y], axis=1)
      z = y

    in_channels, out_channels = self._get_in_out_channels()
    num_blocks = len(in_channels)

    # Map noise to the actual seed.
    net = ops.linear(
        z,
        in_channels[0] * seed_size * seed_size,
        scope="fc_noise",
        use_sn=self._spectral_norm)
    # Reshape the seed to be a rank-4 Tensor.
    net = tf.reshape(
        net,
        [-1, seed_size, seed_size, in_channels[0]],
        name="fc_reshaped")

    for block_idx in range(num_blocks):
      scale = "none" if block_idx % 2 == 0 else "up"
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale=scale)
      net = block(net, z=z, y=y, is_training=is_training)
      # At resolution 64x64 there is a self-attention block.
      if scale == "up" and net.shape[1].value == 64:
        logging.info("[Generator] Applying non-local block to %s", net.shape)
        net = ops.non_local_block(net, "non_local_block",
                                  use_sn=self._spectral_norm)
    # Final processing of the net.
    # Use unconditional batch norm.
    logging.info("[Generator] before final processing: %s", net.shape)
    net = ops.batch_norm(net, is_training=is_training, name="final_norm")
    net = tf.nn.relu(net)
    colors = self._image_shape[2]
    if self._experimental_fast_conv_to_rgb:

      net = ops.conv2d(net, output_dim=128, k_h=3, k_w=3,
                       d_h=1, d_w=1, name="final_conv",
                       use_sn=self._spectral_norm)
      net = net[:, :, :, :colors]
    else:
      net = ops.conv2d(net, output_dim=colors, k_h=3, k_w=3,
                       d_h=1, d_w=1, name="final_conv",
                       use_sn=self._spectral_norm)
    logging.info("[Generator] after final processing: %s", net.shape)
    net = (tf.nn.tanh(net) + 1.0) / 2.0
    return net


@gin.configurable
class Discriminator(abstract_arch.AbstractDiscriminator):
  """ResNet-based discriminator supporting resolutions 32, 64, 128, 256, 512."""

  def __init__(self,
               ch=128,
               blocks_with_attention="B1",
               project_y=True,
               **kwargs):
    """Constructor for BigGAN discriminator.

    Args:
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      project_y: Add an embedding of y in the output layer.
      **kwargs: additional arguments past on to ResNetDiscriminator.
    """
    super(Discriminator, self).__init__(**kwargs)
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._project_y = project_y

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return BigGanDeepResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)

  def _get_in_out_channels(self, colors, resolution):
    # See Table 7-9.
    if colors not in [1, 3]:
      raise ValueError("Unsupported color channels: {}".format(colors))
    if resolution == 512:
      channel_multipliers = [1, 1, 1, 2, 2, 4, 4] + 4 * [8] + 4 * [16]
    elif resolution == 256:
      channel_multipliers = [1, 2, 2, 4, 4] + 4 * [8] + 4 * [16]
    elif resolution == 128:
      channel_multipliers = [1, 2, 2, 4, 4] + 2 * [8] + 4 * [16]
    elif resolution == 64:
      channel_multipliers = [2, 4, 4] + 2 * [8] + 4 * [16]
    elif resolution == 32:
      channel_multipliers = 8 * [2]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self._ch * c for c in channel_multipliers[:-1]]
    out_channels = [self._ch * c for c in channel_multipliers[1:]]
    return in_channels, out_channels

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
    logging.info("[Discriminator] inputs are x=%s, y=%s", x.shape,
                 None if y is None else y.shape)
    resnet_ops.validate_image_inputs(x)

    in_channels, out_channels = self._get_in_out_channels(
        colors=x.shape[-1].value, resolution=x.shape[1].value)
    num_blocks = len(in_channels)

    net = ops.conv2d(x, output_dim=in_channels[0], k_h=3, k_w=3,
                     d_h=1, d_w=1, name="initial_conv",
                     use_sn=self._spectral_norm)

    for block_idx in range(num_blocks):
      scale = "down" if block_idx % 2 == 0 else "none"
      block = self._resnet_block(
          name="B{}".format(block_idx + 1),
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale=scale)
      net = block(net, z=None, y=y, is_training=is_training)
      # At resolution 64x64 there is a self-attention block.
      if scale == "none" and net.shape[1].value == 64:
        logging.info("[Discriminator] Applying non-local block to %s",
                     net.shape)
        net = ops.non_local_block(net, "non_local_block",
                                  use_sn=self._spectral_norm)

    # Final part
    logging.info("[Discriminator] before final processing: %s", net.shape)
    net = tf.nn.relu(net)
    h = tf.math.reduce_sum(net, axis=[1, 2])
    out_logit = ops.linear(h, 1, scope="final_fc", use_sn=self._spectral_norm)
    logging.info("[Discriminator] after final processing: %s", net.shape)
    if self._project_y:
      if y is None:
        raise ValueError("You must provide class information y to project.")
      with tf.variable_scope("embedding_fc"):
        y_embedding_dim = out_channels[-1]
        # We do not use ops.linear() below since it does not have an option to
        # override the initializer.
        kernel = tf.get_variable(
            "kernel", [y.shape[1], y_embedding_dim], tf.float32,
            initializer=tf.initializers.glorot_normal())
        if self._spectral_norm:
          kernel = ops.spectral_norm(kernel)
        embedded_y = tf.matmul(y, kernel)
        logging.info("[Discriminator] embedded_y for projection: %s",
                     embedded_y.shape)
        out_logit += tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, h
