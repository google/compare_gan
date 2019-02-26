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

"""Interface for GAN models that can be trained using the Estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class AbstractGAN(object):
  """Interface for GAN models that can be training using the Estimator API."""

  def __init__(self,
               dataset,
               parameters,
               model_dir):
    super(AbstractGAN, self).__init__()
    self._dataset = dataset
    self._parameters = parameters
    self._model_dir = model_dir

  def as_estimator(self, run_config, batch_size, use_tpu):
    """Returns a TPUEstimator for this GAN."""
    return tf.contrib.tpu.TPUEstimator(
        config=run_config,
        use_tpu=use_tpu,
        model_fn=self.model_fn,
        train_batch_size=batch_size)

  @abc.abstractmethod
  def as_module_spec(self, params, mode):
    """Returns the generator network as TFHub module spec."""

  @abc.abstractmethod
  def input_fn(self, params, mode):
    """Input function that retuns a `tf.data.Dataset` object.

    This function will be called once for each host machine.

    Args:
      params: Python dictionary with parameters given to TPUEstimator.
          Additional TPUEstimator will set the key `batch_size` with the batch
          size for this host machine and `tpu_contextu` with a TPUContext
          object.
      mode: `tf.estimator.MoedeKeys` value.

    Returns:
      A `tf.data.Dataset` object with batched features and labels.
    """

  @abc.abstractmethod
  def model_fn(self, features, labels, params, mode):
    """Constructs the model for the given features and mode.

    This interface only requires implementing the TRAIN mode.

    On TPUs the model_fn should construct a graph for a single TPU core.
    Wrap the optimizer with a `tf.contrib.tpu.CrossShardOptimizer` to do
    synchronous training with all TPU cores.c

    Args:
      features: A dictionary with the feature tensors.
      labels: Tensor will labels. Will be None if mode is PREDICT.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      mode: `tf.estimator.ModeKeys` value (TRAIN, EVAL, PREDICT). The mode
          should be passed to the TPUEstimatorSpec and your model should be
          build this mode.

    Returns:
      A `tf.contrib.tpu.TPUEstimatorSpec`.
    """
