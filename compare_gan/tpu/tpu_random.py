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

"""Provide methods for generating deterministic pseudorandom values.

Random number generators in `tf.random` ignore the seed values on TPUs.
An alternative are the stateless random number generators in
`tf.contrib.stateless` which are deterministic but do not keep the a state.
To get different but reproducible random values at each training step the user
needs to provide a seed (a tensor of shape (2,)) that should change in every
step.

This small library handles this for the user by decomposing the seed into two
values: a per operation seed and a global offset
The per operation seed is fixed for each random generator in the graph and
computed from the name of the operation (incl. name scope).
The global offset is passed in as in integer from in the input function and
thus changes every step. This guarantees that is different every step and
always different between TPU cores within a step.

Usage:
- In your `input_fn` call `add_random_offset_to_features` and use the
   returned dataset.
- At the beginning of your `model_fn` call `set_random_offset_from_features`.
- Use the random number generators defined in this module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib

from absl import logging
import tensorflow as tf


_RANDOM_OFFSET_FEATURE_KEY = "_RANDOM_OFFSET"
_RANDOM_OFFSET_TENSOR = None


def add_random_offset_to_features(dataset, start=1):
  """Add a random offset to the dataset.

  Args:
    dataset: `tf.data.Dataset` object that contains tuples (features, labels),
        where `features` is a Python dictionary.
    start: A starting value for the global offset. Optional.

  Returns:
    A new `tf.data.Dataset` object with a extra feature for the random offset.
  """
  dataset = dataset.apply(tf.data.experimental.enumerate_dataset(start=start))
  def map_fn(offset, data):
    offset = tf.cast(offset, tf.int32)
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], dict):
      # Data is a tuple (features, labels) as expected by the Estimator
      # interface.
      logging.info("Passing random offset: %s with data %s.", offset, data)
      features, labels = data
      features[_RANDOM_OFFSET_FEATURE_KEY] = offset
      return features, labels
    raise ValueError("Data in dataset must be a tuple (features, labels) and "
                     "features must be a Python dictionary. data was {}".format(
                         data))
  return dataset.map(map_fn)


def set_random_offset_from_features(features):
  """Set the global random offset from the random offset feature."""
  # Take the first index in case the TPU core got multiple examples.
  global _RANDOM_OFFSET_TENSOR
  _RANDOM_OFFSET_TENSOR = features.pop(_RANDOM_OFFSET_FEATURE_KEY)[0]
  logging.info("Got global random offset: %s", _RANDOM_OFFSET_TENSOR)


def _get_seed(name=None):
  """Get a deterministic random seed for stateless generators.

  Args:
    name: Name of the operation that will use the seed. If None a unique name
        will be determined.

  Returns:
    An integer`Tensor` of shape (2,) with the seed for this op and the global
    random offset.
  """
  if _RANDOM_OFFSET_TENSOR is None:
    raise ValueError("_RANDOM_OFFSET_TENSOR is None. Did you call "
                     "set_random_offset_from_features() in your model_fn?")
  # Get a seed from the hash name of a dummy operation. This seed will only
  # depend on the name of the operation (incl. the scope name). It will be
  # unique within the graph and only change if the name of operation changes.
  with tf.name_scope("dummy_for_seed"):
    dummy_op = tf.no_op(name)
  # Using SHA-512 gives us a non-negative and uniformly distributed seed in the
  # interval [0, 2**512). This is consistent with TensorFlow, as TensorFlow
  # operations internally use the residue of the given seed modulo `2**31 - 1`
  # (see`tensorflow/python/framework/random_seed.py`).
  op_seed = int(hashlib.sha512(dummy_op.name.encode("utf-8")).hexdigest(), 16)
  op_seed = tf.constant(op_seed % (2**31 - 1))
  logging.info("Using op_seed %s for operation %s.", op_seed, dummy_op.name)
  return tf.stack([op_seed, _RANDOM_OFFSET_TENSOR])


def uniform(shape, name=None):
  """Outputs pseudorandom random values from a uniform distribution.

  If the _RANDOM_OFFSET_TENSOR is set these output is deterministic based on the
  seed and the `name` of this operation. If `name` is None this will use the
  index in the graph instead.

  There is no `dtype` parameter since the underlying
  tf.contrib.stateless.stateless_random_uniform only supports tf.half,
  tf.float32 and tf.float64 and we do not care about tf.half and tf.float64.
  Patches welcome.

  Args:
    shape: A Tensor. Must be one of the following types: int32, int64.
        The shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A Tensor.
  """
  if _RANDOM_OFFSET_TENSOR is None:
    logging.warning("No global random offset set, falling back to "
                    "un-deterministic pseudorandom numbers for operation %s.",
                    name)
    return tf.random.uniform(shape, name=name)
  return tf.contrib.stateless.stateless_random_uniform(
      shape=shape, seed=_get_seed(name), name=name)


def normal(shape, name=None):
  if _RANDOM_OFFSET_TENSOR is None:
    logging.warning("No global random offset set, falling back to "
                    "un-deterministic pseudorandom numbers for operation %s.",
                    name)
    return tf.random.normal(shape, name=name)
  return tf.contrib.stateless.stateless_random_normal(
      shape=shape, seed=_get_seed(name), name=name)
