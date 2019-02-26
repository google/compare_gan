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

"""Tensorflow operations specific to TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
from six.moves import range
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function


def cross_replica_concat(value, replica_id, num_replicas):
  """Reduce a concatenation of the `value` across TPU replicas.

  Args:
    value: Tensor to concatenate.
    replica_id: Integer tensor that indicates the index of the replica.
    num_replicas: Python integer, total number of replicas.

  Returns:
    Tensor of the same rank as value with first dimension `num_replicas`
    times larger.

  Raises:
    ValueError: If `value` is a scalar.
  """
  if value.shape.ndims < 1:
    raise ValueError("Value must have at least rank 1 but got {}.".format(
        value.shape.ndims))
  if num_replicas <= 1:
    return value
  with tf.name_scope(None, "tpu_cross_replica_concat"):
    # Mask is one hot encoded position of the core_index.
    mask = tf.to_float(tf.equal(tf.range(num_replicas), replica_id))
    # Expand dims with 1's to match rank of value.
    mask = tf.reshape(mask, [num_replicas] + [1] * value.shape.ndims)
    if value.dtype in {tf.bfloat16, tf.float32}:
      result = mask * value
    else:
      result = mask * tf.to_float(value)
    # Thanks to broadcasting now result is set only in the position pointed by
    # replica_id, the rest of the vector is set to 0's.
    # All these steps are basically implementing tf.scatter_nd which is missing
    # in TPU's backend since it doesn't support sparse operations.

    # Merge first 2 dimensions.
    # This is equivalent to (value.shape[0].value * num_replicas).
    # Using [-1] trick to support also scalar input.
    result = tf.reshape(result, [-1] + result.shape.as_list()[2:])
    # Each core set the "results" in position pointed by replica_id. When we now
    # sum across replicas we exchange the information and fill in local 0's with
    # values from other cores.
    result = tf.contrib.tpu.cross_replica_sum(result)
    # Now all the cores see exactly the same data.
    return tf.cast(result, dtype=value.dtype)


def cross_replica_mean(inputs, group_size=None):
  """Calculates the average value of inputs tensor across TPU replicas."""
  num_replicas = tpu_function.get_tpu_context().number_of_shards
  if not group_size:
    group_size = num_replicas
  if group_size == 1:
    return inputs
  if group_size != num_replicas:
    group_assignment = []
    assert num_replicas % group_size == 0
    for g in range(num_replicas // group_size):
      replica_ids = [g * group_size + i for i in range(group_size)]
      group_assignment.append(replica_ids)
  else:
    group_assignment = None
  return tf.contrib.tpu.cross_replica_sum(inputs, group_assignment) / tf.cast(
      group_size, inputs.dtype)


@gin.configurable(blacklist=["inputs", "axis"])
def cross_replica_moments(inputs, axis, parallel=True, group_size=None):
  """Compute mean and variance of the inputs tensor across TPU replicas.

  Args:
    inputs: A tensor with 2 or more dimensions.
    axis: Array of ints. Axes along which to compute mean and variance.
    parallel: Use E[x^2] - (E[x])^2 to compute variance. Then can be done
      in parallel to computing the mean and reducing the communication overhead.
    group_size: Integer, the number of replicas to compute moments arcoss.
      None or 0 will use all replicas (global).

  Returns:
    Two tensors with mean and variance.
  """
  # Compute local mean and then average across replicas.
  mean = tf.math.reduce_mean(inputs, axis=axis)
  mean = cross_replica_mean(mean)
  if parallel:
    # Compute variance using the E[x^2] - (E[x])^2 formula. This is less
    # numerically stable than the E[(x-E[x])^2] formula, but allows the two
    # cross-replica sums to be computed in parallel, saving communication
    # overhead.
    mean_of_squares = tf.reduce_mean(tf.square(inputs), axis=axis)
    mean_of_squares = cross_replica_mean(mean_of_squares, group_size=group_size)
    mean_squared = tf.square(mean)
    variance = mean_of_squares - mean_squared
  else:
    variance = tf.math.reduce_mean(
        tf.math.square(inputs - mean), axis=axis)
  variance = cross_replica_mean(variance, group_size=group_size)
  return mean, variance
