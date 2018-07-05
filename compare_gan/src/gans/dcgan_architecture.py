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

"""Implementation of DCGAN generator and discriminator architectures."""
from __future__ import division

from compare_gan.src.gans import consts
from compare_gan.src.gans.ops import lrelu, linear, conv2d, deconv2d

import numpy as np
import tensorflow as tf


def batch_norm_dcgan(input_, is_training, scope, decay=0.9, epsilon=1e-5):
  return tf.contrib.layers.batch_norm(
      input_,
      decay=decay,
      updates_collections=None,
      epsilon=epsilon,
      scale=True,
      fused=False,  # Interesting.
      is_training=is_training,
      scope=scope)


def conv_out_size_same(size, stride):
  return int(np.ceil(float(size) / float(stride)))


def discriminator(x,
                  batch_size,
                  is_training,
                  discriminator_normalization,
                  reuse=False):
  """Returns the outputs of the DCGAN discriminator.

  Details are available at https://arxiv.org/abs/1511.06434. Notable changes
  include BatchNorm in the discriminator and LeakyReLU for all layers.

  Args:
    x: input images, shape [bs, h, w, channels].
    batch_size: integer, number of samples in batch.
    is_training: boolean, are we in train or eval model.
    discriminator_normalization: which type of normalization to apply.
    reuse: boolean, should params be re-used.

  Returns:
    out: A float (in [0, 1]) with discriminator prediction.
    out_logit: Logits (activations of the last linear layer).
    net: Logits of the last ReLu layer.
  """
  assert discriminator_normalization in [
      consts.NO_NORMALIZATION, consts.SPECTRAL_NORM, consts.BATCH_NORM]
  bs = batch_size
  df_dim = 64  # Dimension of filters in first convolutional layer.
  use_sn = discriminator_normalization == consts.SPECTRAL_NORM
  with tf.variable_scope("discriminator", reuse=reuse):
    net = lrelu(conv2d(x, df_dim, 5, 5, 2, 2, name="d_conv1", use_sn=use_sn))
    net = conv2d(net, df_dim * 2, 5, 5, 2, 2, name="d_conv2", use_sn=use_sn)

    if discriminator_normalization == consts.BATCH_NORM:
      net = batch_norm_dcgan(net, is_training, scope="d_bn1")
    net = lrelu(net)
    net = conv2d(net, df_dim * 4, 5, 5, 2, 2, name="d_conv3", use_sn=use_sn)

    if discriminator_normalization == consts.BATCH_NORM:
      net = batch_norm_dcgan(net, is_training, scope="d_bn2")
    net = lrelu(net)
    net = conv2d(net, df_dim * 8, 5, 5, 2, 2, name="d_conv4", use_sn=use_sn)

    if discriminator_normalization == consts.BATCH_NORM:
      net = batch_norm_dcgan(net, is_training, scope="d_bn3")
    net = lrelu(net)
    out_logit = linear(
        tf.reshape(net, [bs, -1]), 1, scope="d_fc4", use_sn=use_sn)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, net


def generator(z, batch_size, output_height, output_width, output_c_dim,
              is_training, reuse=False):
  """Returns the output tensor of the DCGAN generator.

  Details are available at https://arxiv.org/abs/1511.06434. Notable changes
  include BatchNorm in the generator, ReLu instead of LeakyReLu and ReLu in
  generator, except for output which uses TanH.

  Args:
    z: latent code, shape [batch_size, latent_dimensionality]
    batch_size: Batch size.
    output_height: Output image height.
    output_width: Output image width.
    output_c_dim: Number of color channels.
    is_training: boolean, are we in train or eval model.
    reuse: boolean, should params be re-used.

  Returns:
    net: The generated image Tensor with entries in [0, 1].
  """
  gf_dim = 64  # Dimension of filters in first convolutional layer.
  bs = batch_size
  with tf.variable_scope("generator", reuse=reuse):
    s_h, s_w = output_height, output_width
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    net = linear(z, gf_dim * 8 *s_h16 * s_w16, scope="g_fc1")
    net = tf.reshape(net, [-1, s_h16, s_w16, gf_dim * 8])
    net = tf.nn.relu(batch_norm_dcgan(net, is_training, scope="g_bn1"))
    net = deconv2d(net, [bs, s_h8, s_w8, gf_dim*4], 5, 5, 2, 2, name="g_dc1")
    net = tf.nn.relu(batch_norm_dcgan(net, is_training, scope="g_bn2"))
    net = deconv2d(net, [bs, s_h4, s_w4, gf_dim*2], 5, 5, 2, 2, name="g_dc2")
    net = tf.nn.relu(batch_norm_dcgan(net, is_training, scope="g_bn3"))
    net = deconv2d(net, [bs, s_h2, s_w2, gf_dim*1], 5, 5, 2, 2, name="g_dc3")
    net = tf.nn.relu(batch_norm_dcgan(net, is_training, scope="g_bn4"))
    net = deconv2d(net, [bs, s_h, s_w, output_c_dim], 5, 5, 2, 2, name="g_dc4")
    net = 0.5 * tf.nn.tanh(net) + 0.5
    return net


def sn_discriminator(x, batch_size, reuse=False, use_sn=False):
  """Returns the outputs of the SNDCGAN discriminator.

  Details are available at https://openreview.net/pdf?id=B1QRgziT-.

  Args:
    x: input images, shape [bs, h, w, channels].
    batch_size: integer, number of samples in batch.
    reuse: boolean, should params be re-used.

  Returns:
    out: A float (in [0, 1]) with discriminator prediction.
    out_logit: Logits (activations of the last linear layer).
    net: Logits of the last ReLu layer.
  """

  # In compare gan framework, the image preprocess normalize image pixel to
  # range [0, 1], while author used [-1, 1]. Apply this trick to input image
  # instead of changing our preprocessing function.
  x = x * 2.0 - 1.0
  with tf.variable_scope("discriminator", reuse=reuse):
    # Mapping x from [bs, h, w, c] to [bs, 1]
    normal = tf.random_normal_initializer
    net = conv2d(
        x, 64, 3, 3, 1, 1, name="d_conv1", initializer=normal, use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(
        net, 128, 4, 4, 2, 2, name="d_conv2", initializer=normal, use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(
        net, 128, 3, 3, 1, 1, name="d_conv3", initializer=normal, use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(
        net, 256, 4, 4, 2, 2, name="d_conv4", initializer=normal, use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(
        net, 256, 3, 3, 1, 1, name="d_conv5", initializer=normal, use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(
        net, 512, 4, 4, 2, 2, name="d_conv6", initializer=normal, use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(
        net, 512, 3, 3, 1, 1, name="d_conv7", initializer=normal, use_sn=use_sn)
    net = lrelu(net, leak=0.1)

    net = tf.reshape(net, [batch_size, -1])
    out_logit = linear(net, 1, scope="d_fc1", use_sn=use_sn)
    out_logit = tf.squeeze(out_logit)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, net


def sn_generator(z,
                 batch_size,
                 output_height,
                 output_width,
                 output_c_dim,
                 is_training,
                 reuse=False):
  """Returns the output tensor of the SNDCGAN generator.

  Details are available at https://openreview.net/pdf?id=B1QRgziT-.

  Args:
    z: latent code, shape [batch_size, latent_dimensionality]
    batch_size: Batch size.
    output_height: Output image height.
    output_width: Output image width.
    output_c_dim: Number of color channels.
    is_training: boolean, are we in train or eval model.
    reuse: boolean, should params be re-used.

  Returns:
    net: The generated image Tensor with entries in [0, 1].
  """
  s_h, s_w = output_height, output_width
  s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
  s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
  s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

  with tf.variable_scope("generator", reuse=reuse):
    net = linear(z, s_h8 * s_w8 * 512, scope="g_fc1")
    net = batch_norm_dcgan(net, is_training, scope="g_bn1", epsilon=2e-5)
    net = tf.nn.relu(net)
    net = tf.reshape(net, [batch_size, s_h8, s_w8, 512])
    net = deconv2d(net, [batch_size, s_h4, s_w4, 256], 4, 4, 2, 2, name="g_dc2")
    net = batch_norm_dcgan(net, is_training, scope="g_bn2", epsilon=2e-5)
    net = tf.nn.relu(net)
    net = deconv2d(net, [batch_size, s_h2, s_w2, 128], 4, 4, 2, 2, name="g_dc3")
    net = batch_norm_dcgan(net, is_training, scope="g_bn3", epsilon=2e-5)
    net = tf.nn.relu(net)
    net = deconv2d(net, [batch_size, s_h, s_w, 64], 4, 4, 2, 2, name="g_dc4")
    net = batch_norm_dcgan(net, is_training, scope="g_bn4", epsilon=2e-5)
    net = tf.nn.relu(net)
    net = deconv2d(
        net, [batch_size, s_h, s_w, output_c_dim], 3, 3, 1, 1, name="g_dc5")
    out = tf.tanh(net)

    # NOTE: this normalization is introduced to match current image
    # preprocessing, which normalize the real image to range [0, 1].
    # In author's implementation, they simply use the tanh activation function
    # and normalize the image to range [-1, 1].
    out = tf.div(out + 1.0, 2.0)

    return out
