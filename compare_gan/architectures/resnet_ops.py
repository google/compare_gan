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

"""ResNet specific operations.

Defines the default ResNet generator and discriminator blocks and some helper
operations such as unpooling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from compare_gan.architectures import abstract_arch
from compare_gan.architectures import arch_ops as ops

from six.moves import range
import tensorflow as tf


def unpool(value, name="unpool"):
  """Unpooling operation.

  N-dimensional version of the unpooling operation from
  https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
  Taken from: https://github.com/tensorflow/tensorflow/issues/2169

  Args:
    value: a Tensor of shape [b, d0, d1, ..., dn, ch]
    name: name of the op
  Returns:
    A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
  """
  with tf.name_scope(name) as scope:
    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
      out = tf.concat([out, tf.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size, name=scope)
  return out


def validate_image_inputs(inputs, validate_power2=True):
  inputs.get_shape().assert_has_rank(4)
  inputs.get_shape()[1:3].assert_is_fully_defined()
  if inputs.get_shape()[1] != inputs.get_shape()[2]:
    raise ValueError("Input tensor does not have equal width and height: ",
                     inputs.get_shape()[1:3])
  width = inputs.get_shape().as_list()[1]
  if validate_power2 and math.log(width, 2) != int(math.log(width, 2)):
    raise ValueError("Input tensor `width` is not a power of 2: ", width)


class ResNetBlock(object):
  """ResNet block with options for various normalizations."""

  def __init__(self,
               name,
               in_channels,
               out_channels,
               scale,
               is_gen_block,
               layer_norm=False,
               spectral_norm=False,
               batch_norm=None):
    """Constructs a new ResNet block.

    Args:
      name: Scope name for the resent block.
      in_channels: Integer, the input channel size.
      out_channels: Integer, the output channel size.
      scale: Whether or not to scale up or down, choose from "up", "down" or
        "none".
      is_gen_block: Boolean, deciding whether this is a generator or
        discriminator block.
      layer_norm: Apply layer norm before both convolutions.
      spectral_norm: Use spectral normalization for all weights.
      batch_norm: Function for batch normalization.
    """
    assert scale in ["up", "down", "none"]
    self._name = name
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._scale = scale
    # In SN paper, if they upscale in generator they do this in the first conv.
    # For discriminator downsampling happens after second conv.
    self._scale1 = scale if is_gen_block else "none"
    self._scale2 = "none" if is_gen_block else scale
    self._layer_norm = layer_norm
    self._spectral_norm = spectral_norm
    self.batch_norm = batch_norm

  def __call__(self, inputs, z, y, is_training):
    return self.apply(inputs=inputs, z=z, y=y, is_training=is_training)

  def _get_conv(self, inputs, in_channels, out_channels, scale, suffix,
                kernel_size=(3, 3), strides=(1, 1)):
    """Performs a convolution in the ResNet block."""
    if inputs.get_shape().as_list()[-1] != in_channels:
      raise ValueError("Unexpected number of input channels.")
    if scale not in ["up", "down", "none"]:
      raise ValueError(
          "Scale: got {}, expected 'up', 'down', or 'none'.".format(scale))

    outputs = inputs
    if scale == "up":
      outputs = unpool(outputs)
    outputs = ops.conv2d(
        outputs,
        output_dim=out_channels,
        k_h=kernel_size[0], k_w=kernel_size[1],
        d_h=strides[0], d_w=strides[1],
        use_sn=self._spectral_norm,
        name="{}_{}".format("same" if scale == "none" else scale, suffix))
    if scale == "down":
      outputs = tf.nn.pool(outputs, [2, 2], "AVG", "SAME", strides=[2, 2],
                           name="pool_%s" % suffix)
    return outputs

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
    if inputs.get_shape().as_list()[-1] != self._in_channels:
      raise ValueError("Unexpected number of input channels.")

    with tf.variable_scope(self._name, values=[inputs]):
      output = inputs

      shortcut = self._get_conv(
          output, self._in_channels, self._out_channels, self._scale,
          suffix="conv_shortcut")

      output = self.batch_norm(
          output, z=z, y=y, is_training=is_training, name="bn1")
      if self._layer_norm:
        output = ops.layer_norm(output, is_training=is_training, scope="ln1")

      output = tf.nn.relu(output)
      output = self._get_conv(
          output, self._in_channels, self._out_channels, self._scale1,
          suffix="conv1")

      output = self.batch_norm(
          output, z=z, y=y, is_training=is_training, name="bn2")
      if self._layer_norm:
        output = ops.layer_norm(output, is_training=is_training, scope="ln2")

      output = tf.nn.relu(output)
      output = self._get_conv(
          output, self._out_channels, self._out_channels, self._scale2,
          suffix="conv2")

      # Combine skip-connection with the convolved part.
      output += shortcut
      return output


class ResNetGenerator(abstract_arch.AbstractGenerator):
  """Abstract base class for generators based on the ResNet architecture."""

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return ResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=True,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)


class ResNetDiscriminator(abstract_arch.AbstractDiscriminator):
  """Abstract base class for discriminators based on the ResNet architecture."""

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return ResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=False,
        layer_norm=self._layer_norm,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)
