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

"""Defines interfaces for generator and discriminator networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from compare_gan import utils
import gin
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class _Module(object):
  """Base class for architectures.

  Long term this will be replaced by `tf.Module` in TF 2.0.
  """

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def trainable_variables(self):
    return [var for var in tf.trainable_variables() if self._name in var.name]


@gin.configurable("G", blacklist=["name", "image_shape"])
class AbstractGenerator(_Module):
  """Interface for generator architectures."""

  def __init__(self,
               name="generator",
               image_shape=None,
               batch_norm_fn=None,
               spectral_norm=False):
    """Constructor for all generator architectures.

    Args:
      name: Scope name of the generator.
      image_shape: Image shape to be generated, [height, width, colors].
      batch_norm_fn: Function for batch normalization or None.
      spectral_norm: If True use spectral normalization for all weights.
    """
    super(AbstractGenerator, self).__init__(name=name)
    self._name = name
    self._image_shape = image_shape
    self._batch_norm_fn = batch_norm_fn
    self._spectral_norm = spectral_norm

  def __call__(self, z, y, is_training, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.name, values=[z, y], reuse=reuse):
      outputs = self.apply(z=z, y=y, is_training=is_training)
    return outputs

  def batch_norm(self, inputs, **kwargs):
    if self._batch_norm_fn is None:
      return inputs
    args = kwargs.copy()
    args["inputs"] = inputs
    if "use_sn" not in args:
      args["use_sn"] = self._spectral_norm
    return utils.call_with_accepted_args(self._batch_norm_fn, **args)

  @abc.abstractmethod
  def apply(self, z, y, is_training):
    """Apply the generator on a input.

    Args:
      z: `Tensor` of shape [batch_size, z_dim] with latent code.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: Boolean, whether the architecture should be constructed for
        training or inference.

    Returns:
      Generated images of shape [batch_size] + self.image_shape.
    """


@gin.configurable("D", blacklist=["name"])
class AbstractDiscriminator(_Module):
  """Interface for discriminator architectures."""

  def __init__(self,
               name="discriminator",
               batch_norm_fn=None,
               layer_norm=False,
               spectral_norm=False):
    super(AbstractDiscriminator, self).__init__(name=name)
    self._name = name
    self._batch_norm_fn = batch_norm_fn
    self._layer_norm = layer_norm
    self._spectral_norm = spectral_norm

  def __call__(self, x, y, is_training, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.name, values=[x, y], reuse=reuse):
      outputs = self.apply(x=x, y=y, is_training=is_training)
    return outputs

  def batch_norm(self, inputs, **kwargs):
    if self._batch_norm_fn is None:
      return inputs
    args = kwargs.copy()
    args["inputs"] = inputs
    if "use_sn" not in args:
      args["use_sn"] = self._spectral_norm
    return utils.call_with_accepted_args(self._batch_norm_fn, **args)


  @abc.abstractmethod
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
