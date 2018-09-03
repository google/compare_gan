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

"""This file is only used for ablation study for Resnet.

Don't use this implementation in other experiments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import log

from compare_gan.src.gans import consts
from compare_gan.src.gans import ops

import numpy as np
from six.moves import range
import tensorflow as tf

slim = tf.contrib.slim
tfgan = tf.contrib.gan


def _validate_image_inputs(inputs, validate_power2=True):
  inputs.get_shape().assert_has_rank(4)
  inputs.get_shape()[1:3].assert_is_fully_defined()
  if inputs.get_shape()[1] != inputs.get_shape()[2]:
    raise ValueError("Input tensor does not have equal width and height: ",
                     inputs.get_shape()[1:3])
  width = inputs.get_shape().as_list()[1]
  if validate_power2 and log(width, 2) != int(log(width, 2)):
    raise ValueError("Input tensor `width` is not a power of 2: ", width)


def batch_norm_resnet(input_, is_training, scope, epsilon=1e-5):
  return tf.contrib.layers.batch_norm(
      input_,
      decay=0.9,
      updates_collections=None,
      epsilon=epsilon,
      scale=True,
      fused=False,  # Interesting.
      is_training=is_training,
      scope=scope)


# From https://github.com/tensorflow/tensorflow/issues/2169
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


def get_conv(inputs, in_channels, out_channels, scale, suffix, use_sn):
  """Return convolution for the resnet block."""
  if inputs.get_shape().as_list()[-1] != in_channels:
    raise ValueError("Unexpected number of input channels.")

  if scale == "up":
    output = unpool(inputs)
    output = ops.conv2d(
        output, output_dim=out_channels, k_h=3, k_w=3,
        d_h=1, d_w=1, name="up_%s" % suffix, use_sn=use_sn)
    return output
  elif scale == "none":
    return ops.conv2d(
        inputs, output_dim=out_channels, k_h=3, k_w=3,
        d_h=1, d_w=1, name="same_%s" % suffix, use_sn=use_sn)
  elif scale == "down":
    output = ops.conv2d(
        inputs, output_dim=out_channels, k_h=3, k_w=3,
        d_h=1, d_w=1, name="down_%s" % suffix, use_sn=use_sn)
    output = tf.nn.pool(
        output, [2, 2], "AVG", "SAME", strides=[2, 2], name="pool_%s" % suffix)
    return output
  return None  # Should not happen!


def resnet_block(inputs, in_channels, out_channels, scale,
                 block_scope, is_training, reuse, discriminator_normalization,
                 is_gen_block):
  assert scale in ["up", "down", "none"]
  if inputs.get_shape().as_list()[-1] != in_channels:
    raise ValueError("Unexpected number of input channels.")

  # In SN paper, if they upscale in generator they do this in the first conv.
  # For discriminator downsampling happens after second conv.
  if is_gen_block:
    # Generator block
    scale1 = scale  # "up" or "none"
    scale2 = "none"
  else:
    # Discriminator block.
    scale1 = "none"
    scale2 = scale  # "down" or "none"

  print ("resnet_block, in=%d out=%d, scale=%s, scope=%s normalizer=%s" % (
      in_channels, out_channels, scale, block_scope,
      discriminator_normalization))
  print ("INPUTS: ", inputs.get_shape())
  with tf.variable_scope(block_scope, values=[inputs], reuse=reuse):
    output = inputs
    use_sn = discriminator_normalization == consts.SPECTRAL_NORM

    # Define the skip connection, ensure 'conv' is in the suffix, otherwise it
    # will not be regularized.

    shortcut = get_conv(
        output, in_channels, out_channels, scale,
        suffix="conv_shortcut", use_sn=use_sn)
    print ("SHORTCUT: ", shortcut.get_shape())

    # Apply batch norm in discriminator only if enabled.
    if is_gen_block or discriminator_normalization == consts.BATCH_NORM:
      output = batch_norm_resnet(output, is_training=is_training, scope="bn1")
    elif discriminator_normalization == consts.LAYER_NORM:
      output = ops.layer_norm(output, is_training=is_training, scope="ln1")

    output = tf.nn.relu(output)
    output = get_conv(
        output, in_channels, out_channels, scale1,
        suffix="conv1", use_sn=use_sn)
    print ("OUTPUT CONV1: ", output.get_shape())

    # Apply batch norm in discriminator only if enabled.
    if is_gen_block or discriminator_normalization == consts.BATCH_NORM:
      output = batch_norm_resnet(output, is_training=is_training, scope="bn2")
    elif discriminator_normalization == consts.LAYER_NORM:
      output = ops.layer_norm(output, is_training=is_training, scope="ln2")

    output = tf.nn.relu(output)
    output = get_conv(
        output, out_channels, out_channels, scale2,
        suffix="conv2", use_sn=use_sn)
    print ("OUTPUT CONV2: ", output.get_shape())

    # Combine skip-connection with the convolved part.
    output += shortcut

    return output


def generator_block(inputs, in_channels, out_channels, scale,
                    block_scope, is_training, reuse):
  assert scale in ["up", "none"]
  return resnet_block(inputs, in_channels, out_channels, scale,
                      block_scope, is_training, reuse,
                      discriminator_normalization=consts.NO_NORMALIZATION,
                      is_gen_block=True)


def discriminator_block(inputs, in_channels, out_channels, scale,
                        block_scope, is_training, reuse,
                        discriminator_normalization):
  assert scale in ["down", "none"]
  return resnet_block(inputs, in_channels, out_channels, scale,
                      block_scope, is_training, reuse,
                      discriminator_normalization, is_gen_block=False)


# Generates resolution 128x128
def resnet5_generator(noise,
                      is_training,
                      reuse=None,
                      colors=3,
                      output_shape=128,
                      unused_ablation_type=""):
  # Input is a noise tensor of shape [bs, z_dim]
  assert len(noise.get_shape().as_list()) == 2

  # Calculate / define a few numbers.
  batch_size = noise.get_shape().as_list()[0]
  # Each block upscales by a factor of 2:
  seed_size = 4
  # We want the last block to have 64 channels:
  ch = 64

  with tf.variable_scope("generator", reuse=reuse):
    # Map noise to the actual seed.
    output = ops.linear(noise, ch * 8 * seed_size * seed_size, scope="fc_noise")

    # Reshape the seed to be a rank-4 Tensor.
    output = tf.reshape(
        output, [batch_size, seed_size, seed_size, ch * 8], name="fc_reshaped")

    # Magic in/out channel numbers copied from SN paper.
    magic = [(8, 8), (8, 4), (4, 4), (4, 2), (2, 1)]
    up_layers = np.log2(float(output_shape) / seed_size)
    assert up_layers.is_integer(), "log2(%d/%d) must be an integer" % (
        output_shape, seed_size)
    assert up_layers <= 5 and up_layers >= 0, "Invalid output_shape %d" % (
        output_shape)
    up_layers = int(up_layers)
    for block_idx in range(5):
      block_scope = "B%d" % (block_idx + 1)
      in_channels = ch * magic[block_idx][0]
      out_channels = ch * magic[block_idx][1]
      print("Resnet5, block %d in=%d out=%d" % (block_idx, in_channels,
                                                out_channels))
      if block_idx < up_layers:
        scale = "up"
      else:
        scale = "none"
      output = generator_block(output, in_channels=in_channels,
                               out_channels=out_channels,
                               scale=scale, block_scope=block_scope,
                               is_training=is_training, reuse=reuse)

    # Final processing of the output.
    output = batch_norm_resnet(output, is_training=is_training,
                               scope="final_norm")
    output = tf.nn.relu(output)
    output = ops.conv2d(
        output, output_dim=colors, k_h=3, k_w=3, d_h=1, d_w=1,
        name="final_conv")
    output = tf.nn.sigmoid(output)

    print("Generator output shape: ", output)
    return output


def resnet5_discriminator(inputs,
                          is_training,
                          discriminator_normalization,
                          reuse=None,
                          unused_ablation_type=""):
  """ResNet style discriminator.

  Construct discriminator network from inputs to the final endpoint.

  Args:
    inputs: A tensor of size [batch_size, height, width, channels]. Must be
      floating point.
    is_training: Is the model currently being trained.
    discriminator_normalization: which type of normalization to apply.
    reuse: Whether or not the network variables should be reused. `scope`
      must be given to be reused.
    unused_ablation_type: str, which type of ablation to apply.

  Returns:
    out: The prediction of the discrminator (in [0, 1]). Shape: [bs, 1]
    out_logit: The pre-softmax activations for discrimination
    real/generated, a tensor of size [batch_size, 1]

  Raises:
    ValueError: If the input image shape is not 4-dimensional, if the spatial
      dimensions aren't defined at graph construction time, if the spatial
      dimensions aren't square, or if the spatial dimensions aren"t a power of
      two.
  """

  _validate_image_inputs(inputs)
  colors = inputs.get_shape().as_list()[-1]
  assert colors in [1, 3]

  ch = 64
  with tf.variable_scope("discriminator", values=[inputs], reuse=reuse):
    output = discriminator_block(
        inputs, in_channels=colors, out_channels=ch,
        scale="down", block_scope="B0", is_training=is_training, reuse=reuse,
        discriminator_normalization=discriminator_normalization)

    # Magic in/out channel numbers copied from SN paper.
    magic = [(1, 2), (2, 4), (4, 4), (4, 8), (8, 8)]
    for block_idx in range(5):
      block_scope = "B%d" % (block_idx + 1)
      in_channels = ch * magic[block_idx][0]
      out_channels = ch * magic[block_idx][1]
      print ("Resnet5 disc, block %d in=%d out=%d" % (
          block_idx, in_channels, out_channels))
      output = discriminator_block(
          output, in_channels=in_channels, out_channels=out_channels,
          scale="down", block_scope=block_scope, is_training=is_training,
          reuse=reuse, discriminator_normalization=discriminator_normalization)

    # Final part
    output = tf.nn.relu(output)
    pre_logits = tf.reduce_mean(output, axis=[1, 2])

    use_sn = discriminator_normalization == consts.SPECTRAL_NORM
    out_logit = ops.linear(pre_logits, 1, scope="disc_final_fc", use_sn=use_sn)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, None


