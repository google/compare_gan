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

"""Implementation of DCGAN generator and discriminator architectures.

Details are available in https://arxiv.org/abs/1511.06434.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import abstract_arch
from compare_gan.architectures.arch_ops import conv2d
from compare_gan.architectures.arch_ops import deconv2d
from compare_gan.architectures.arch_ops import linear
from compare_gan.architectures.arch_ops import lrelu

import numpy as np
import tensorflow as tf


def conv_out_size_same(size, stride):
  return int(np.ceil(float(size) / float(stride)))


class Generator(abstract_arch.AbstractGenerator):
  """DCGAN generator.

  Details are available at https://arxiv.org/abs/1511.06434. Notable changes
  include BatchNorm in the generator, ReLu instead of LeakyReLu and ReLu in the
  generator, except for output which uses tanh.
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
    gf_dim = 64  # Dimension of filters in first convolutional layer.
    bs = z.shape[0].value
    s_h, s_w, colors = self._image_shape
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    net = linear(z, gf_dim * 8 *s_h16 * s_w16, scope="g_fc1")
    net = tf.reshape(net, [-1, s_h16, s_w16, gf_dim * 8])
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn1")
    net = tf.nn.relu(net)
    net = deconv2d(net, [bs, s_h8, s_w8, gf_dim*4], 5, 5, 2, 2, name="g_dc1")
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn2")
    net = tf.nn.relu(net)
    net = deconv2d(net, [bs, s_h4, s_w4, gf_dim*2], 5, 5, 2, 2, name="g_dc2")
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn3")
    net = tf.nn.relu(net)
    net = deconv2d(net, [bs, s_h2, s_w2, gf_dim*1], 5, 5, 2, 2, name="g_dc3")
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn4")
    net = tf.nn.relu(net)
    net = deconv2d(net, [bs, s_h, s_w, colors], 5, 5, 2, 2, name="g_dc4")
    net = 0.5 * tf.nn.tanh(net) + 0.5
    return net


class Discriminator(abstract_arch.AbstractDiscriminator):
  """DCGAN discriminator.

  Details are available at https://arxiv.org/abs/1511.06434. Notable changes
  include BatchNorm in the discriminator and LeakyReLU for all layers.
  """

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
    bs = x.shape[0].value
    df_dim = 64  # Dimension of filters in the first convolutional layer.
    net = lrelu(conv2d(x, df_dim, 5, 5, 2, 2, name="d_conv1",
                       use_sn=self._spectral_norm))
    net = conv2d(net, df_dim * 2, 5, 5, 2, 2, name="d_conv2",
                 use_sn=self._spectral_norm)

    net = self.batch_norm(net, y=y, is_training=is_training, name="d_bn1")
    net = lrelu(net)
    net = conv2d(net, df_dim * 4, 5, 5, 2, 2, name="d_conv3",
                 use_sn=self._spectral_norm)

    net = self.batch_norm(net, y=y, is_training=is_training, name="d_bn2")
    net = lrelu(net)
    net = conv2d(net, df_dim * 8, 5, 5, 2, 2, name="d_conv4",
                 use_sn=self._spectral_norm)

    net = self.batch_norm(net, y=y, is_training=is_training, name="d_bn3")
    net = lrelu(net)
    out_logit = linear(
        tf.reshape(net, [bs, -1]), 1, scope="d_fc4", use_sn=self._spectral_norm)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, net
