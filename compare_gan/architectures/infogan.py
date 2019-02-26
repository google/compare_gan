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

"""Implementation of InfoGAN generator and discriminator architectures.

Details are available in https://arxiv.org/pdf/1606.03657.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.architectures import abstract_arch
from compare_gan.architectures.arch_ops import batch_norm
from compare_gan.architectures.arch_ops import conv2d
from compare_gan.architectures.arch_ops import deconv2d
from compare_gan.architectures.arch_ops import linear
from compare_gan.architectures.arch_ops import lrelu

import tensorflow as tf


class Generator(abstract_arch.AbstractGenerator):
  """Generator architecture based on InfoGAN."""

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
    del y
    h, w, c = self._image_shape
    bs = z.shape.as_list()[0]
    net = linear(z, 1024, scope="g_fc1")
    net = lrelu(batch_norm(net, is_training=is_training, name="g_bn1"))
    net = linear(net, 128 * (h // 4) * (w // 4), scope="g_fc2")
    net = lrelu(batch_norm(net, is_training=is_training, name="g_bn2"))
    net = tf.reshape(net, [bs, h // 4, w // 4, 128])
    net = deconv2d(net, [bs, h // 2, w // 2, 64], 4, 4, 2, 2, name="g_dc3")
    net = lrelu(batch_norm(net, is_training=is_training, name="g_bn3"))
    net = deconv2d(net, [bs, h, w, c], 4, 4, 2, 2, name="g_dc4")
    out = tf.nn.sigmoid(net)
    return out


class Discriminator(abstract_arch.AbstractDiscriminator):
  """Discriminator architecture based on InfoGAN."""

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
    use_sn = self._spectral_norm
    batch_size = x.shape.as_list()[0]
    # Resulting shape: [bs, h/2, w/2, 64].
    net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name="d_conv1", use_sn=use_sn))
    # Resulting shape: [bs, h/4, w/4, 128].
    net = conv2d(net, 128, 4, 4, 2, 2, name="d_conv2", use_sn=use_sn)
    net = self.batch_norm(net, y=y, is_training=is_training, name="d_bn2")
    net = lrelu(net)
    # Resulting shape: [bs, h * w * 8].
    net = tf.reshape(net, [batch_size, -1])
    # Resulting shape: [bs, 1024].
    net = linear(net, 1024, scope="d_fc3", use_sn=use_sn)
    net = self.batch_norm(net, y=y, is_training=is_training, name="d_bn3")
    net = lrelu(net)
    # Resulting shape: [bs, 1].
    out_logit = linear(net, 1, scope="d_fc4", use_sn=use_sn)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, net
