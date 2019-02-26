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

"""Implementation of SNDCGAN generator and discriminator architectures."""

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
  """SNDCGAN generator.

  Details are available at https://openreview.net/pdf?id=B1QRgziT-.
  """

  def apply(self, z, y, is_training):
    """Build the generator network for the given inputs.

    Args:
      z: `Tensor` of shape [batch_size, z_dim] with latent code.
      y: `Tensor` of shape [batch_size, num_classes] of one hot encoded labels.
      is_training: boolean, are we in train or eval model.

    Returns:
      A tensor of size [batch_size] + self._image_shape with values in [0, 1].
    """
    batch_size = z.shape[0].value
    s_h, s_w, colors = self._image_shape
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

    net = linear(z, s_h8 * s_w8 * 512, scope="g_fc1")
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn1")
    net = tf.nn.relu(net)
    net = tf.reshape(net, [batch_size, s_h8, s_w8, 512])
    net = deconv2d(net, [batch_size, s_h4, s_w4, 256], 4, 4, 2, 2, name="g_dc2")
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn2")
    net = tf.nn.relu(net)
    net = deconv2d(net, [batch_size, s_h2, s_w2, 128], 4, 4, 2, 2, name="g_dc3")
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn3")
    net = tf.nn.relu(net)
    net = deconv2d(net, [batch_size, s_h, s_w, 64], 4, 4, 2, 2, name="g_dc4")
    net = self.batch_norm(net, z=z, y=y, is_training=is_training, name="g_bn4")
    net = tf.nn.relu(net)
    net = deconv2d(
        net, [batch_size, s_h, s_w, colors], 3, 3, 1, 1, name="g_dc5")
    out = tf.tanh(net)

    # This normalization from [-1, 1] to [0, 1] is introduced for consistency
    # with other models.
    out = tf.div(out + 1.0, 2.0)
    return out


class Discriminator(abstract_arch.AbstractDiscriminator):
  """SNDCGAN discriminator.

  Details are available at https://openreview.net/pdf?id=B1QRgziT-.
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
    del is_training, y
    use_sn = self._spectral_norm
    # In compare gan framework, the image preprocess normalize image pixel to
    # range [0, 1], while author used [-1, 1]. Apply this trick to input image
    # instead of changing our preprocessing function.
    x = x * 2.0 - 1.0
    net = conv2d(x, 64, 3, 3, 1, 1, name="d_conv1", use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(net, 128, 4, 4, 2, 2, name="d_conv2", use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(net, 128, 3, 3, 1, 1, name="d_conv3", use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(net, 256, 4, 4, 2, 2, name="d_conv4", use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(net, 256, 3, 3, 1, 1, name="d_conv5", use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(net, 512, 4, 4, 2, 2, name="d_conv6", use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    net = conv2d(net, 512, 3, 3, 1, 1, name="d_conv7", use_sn=use_sn)
    net = lrelu(net, leak=0.1)
    batch_size = x.shape.as_list()[0]
    net = tf.reshape(net, [batch_size, -1])
    out_logit = linear(net, 1, scope="d_fc1", use_sn=use_sn)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, net
