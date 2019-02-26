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

"""Provides a library of custom architecture-related operations.

It currently provides the following operations:
- linear, conv2d, deconv2d, lrelu
- batch norm, conditional batch norm, self-modulation
- spectral norm, weight norm, layer norm
- self-attention block
- various weight initialization schemes

These operations are supported on both GPUs and TPUs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging

from compare_gan.gans import consts
from compare_gan.tpu import tpu_ops
import gin
from six.moves import range
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.training import moving_averages  # pylint: disable=g-direct-tensorflow-import


@gin.configurable("weights")
def weight_initializer(initializer=consts.NORMAL_INIT, stddev=0.02):
  """Returns the initializer for the given name.

  Args:
    initializer: Name of the initalizer. Use one in consts.INITIALIZERS.
    stddev: Standard deviation passed to initalizer.

  Returns:
    Initializer from `tf.initializers`.
  """
  if initializer == consts.NORMAL_INIT:
    return tf.initializers.random_normal(stddev=stddev)
  if initializer == consts.TRUNCATED_INIT:
    return tf.initializers.truncated_normal(stddev=stddev)
  if initializer == consts.ORTHOGONAL_INIT:
    return tf.initializers.orthogonal()
  raise ValueError("Unknown weight initializer {}.".format(initializer))


def _moving_moments_for_inference(mean, variance, is_training, decay):
  """Use moving averages of moments during inference.

  Args:
    mean: Tensor of shape [num_channels] with the mean of the current batch.
    variance: Tensor of shape [num_channels] with the variance of the current
      batch.
    is_training: Boolean, wheather to construct ops for training or inference
      graph.
    decay: Decay rate to use for moving averages.

  Returns:
    Tuple of (mean, variance) to use. This can the same as the inputs.
  """
  # Create the moving average variables and add them to the appropriate
  # collections.
  variable_collections = [
      tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
      tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,
  ]
  # Disable partition setting for moving_mean and moving_variance
  # as assign_moving_average op below doesn"t support partitioned variable.
  moving_mean = tf.get_variable(
      "moving_mean",
      shape=mean.shape,
      initializer=tf.zeros_initializer(),
      trainable=False,
      partitioner=None,
      collections=variable_collections)
  moving_variance = tf.get_variable(
      "moving_variance",
      shape=variance.shape,
      initializer=tf.ones_initializer(),
      trainable=False,
      partitioner=None,
      collections=variable_collections)
  if is_training:
    logging.debug("Adding update ops for moving averages of mean and variance.")
    # Update variables for mean and variance during training.
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean,
        tf.cast(mean, moving_mean.dtype),
        decay,
        zero_debias=False)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance,
        tf.cast(variance, moving_variance.dtype),
        decay,
        zero_debias=False)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
    return mean, variance
  logging.debug("Using moving mean and variance.")
  return moving_mean, moving_variance


def _accumulated_moments_for_inference(mean, variance, is_training):
  """Use accumulated statistics for moments during inference.

  After training the user is responsible for filling the accumulators with the
  actual values. See _UpdateBnAccumulators() in eval_gan_lib.py for an example.

  Args:
    mean: Tensor of shape [num_channels] with the mean of the current batch.
    variance: Tensor of shape [num_channels] with the variance of the current
      batch.
    is_training: Boolean, wheather to construct ops for training or inference
      graph.

  Returns:
    Tuple of (mean, variance) to use. This can the same as the inputs.
  """
  variable_collections = [
      tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES,
  ]
  with tf.variable_scope("accu", values=[mean, variance]):
    # Create variables for accumulating batch statistic and use them during
    # inference. The ops for filling the accumulators must be created and run
    # before eval. See docstring above.
    accu_mean = tf.get_variable(
        "accu_mean",
        shape=mean.shape,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=variable_collections)
    accu_variance = tf.get_variable(
        "accu_variance",
        shape=variance.shape,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=variable_collections)
    accu_counter = tf.get_variable(
        "accu_counter",
        shape=[],
        initializer=tf.initializers.constant(1e-12),
        trainable=False,
        collections=variable_collections)
    update_accus = tf.get_variable(
        "update_accus",
        shape=[],
        dtype=tf.int32,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=variable_collections)

    mean = tf.identity(mean, "mean")
    variance = tf.identity(variance, "variance")

    if is_training:
      return mean, variance

    logging.debug("Using accumulated moments.")
    # Return the accumulated batch statistics and add current batch statistics
    # to accumulators if update_accus variables equals 1.
    def update_accus_fn():
      return tf.group([
          tf.assign_add(accu_mean, mean),
          tf.assign_add(accu_variance, variance),
          tf.assign_add(accu_counter, 1),
      ])
    dep = tf.cond(
        tf.equal(update_accus, 1),
        update_accus_fn,
        tf.no_op)
    with tf.control_dependencies([dep]):
      return accu_mean / accu_counter, accu_variance / accu_counter


@gin.configurable(whitelist=["decay", "epsilon", "use_cross_replica_mean",
                             "use_moving_averages"])
def standardize_batch(inputs,
                      is_training,
                      decay=0.999,
                      epsilon=1e-3,
                      data_format="NHWC",
                      use_moving_averages=True,
                      use_cross_replica_mean=None):
  """Adds TPU-enabled batch normalization layer.

  This version does not apply trainable scale or offset!
  It normalizes a tensor by mean and variance.

  Details on Batch Normalization can be found in "Batch Normalization:
  Accelerating Deep Network Training by Reducing Internal Covariate Shift",
  Ioffe S. and Szegedy C. 2015 [http://arxiv.org/abs/1502.03167].

  Note #1: This method computes the batch statistic across all TPU replicas,
  thus simulating the true batch norm in the distributed setting. If one wants
  to avoid the cross-replica communication set use_cross_replica_mean=False.

  Note #2: When is_training is True the moving_mean and moving_variance need
  to be updated in each training step. By default, the update_ops are placed
  in `tf.GraphKeys.UPDATE_OPS` and they need to be added as a dependency to
  the `train_op`. For example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  Note #3: Reasonable values for `decay` are close to 1.0, typically in the
  multiple-nines range: 0.999, 0.99, 0.9, etc. Lower the `decay` value (trying
  `decay`=0.9) if model experiences reasonably good training performance but
  poor validation and/or test performance.

  Args:
    inputs: A tensor with 2 or 4 dimensions, where the first dimension is
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC`, and the second dimension if `data_format` is
      `NCHW`.
    is_training: Whether or not the layer is in training mode. In training
      mode it would accumulate the statistics of the moments into the
      `moving_mean` and `moving_variance` using an exponential moving average
      with the given `decay`. When is_training=False, these variables are not
      updated, and the precomputed values are used verbatim.
    decay: Decay for the moving averages. See notes above for reasonable
      values.
    epsilon: Small float added to variance to avoid dividing by zero.
    data_format: Input data format. NHWC or NCHW.
    use_moving_averages: If True keep moving averages of mean and variance that
      are used during inference. Otherwise use accumlators.
    use_cross_replica_mean: If True add operations to do computes batch norm
      statistics across all TPU cores. These ops are not compatible with other
      platforms. The default (None) will only add the operations if running
      on TPU.

  Returns:
    The normalized tensor with the same type and shape as `inputs`.
  """
  if data_format not in {"NCHW", "NHWC"}:
    raise ValueError(
        "Invalid data_format {}. Allowed: NCHW, NHWC.".format(data_format))
  if use_cross_replica_mean is None:
    # Default to global batch norm only on TPUs.
    use_cross_replica_mean = (
        tpu_function.get_tpu_context().number_of_shards is not None)
    logging.debug("Automatically determined use_cross_replica_mean=%s.",
                  use_cross_replica_mean)

  inputs = tf.convert_to_tensor(inputs)
  inputs_dtype = inputs.dtype
  inputs_shape = inputs.get_shape()

  num_channels = inputs.shape[-1].value
  if num_channels is None:
    raise ValueError("`C` dimension must be known but is None")

  inputs_rank = inputs_shape.ndims
  if inputs_rank is None:
    raise ValueError("Inputs %s has undefined rank" % inputs.name)
  elif inputs_rank not in [2, 4]:
    raise ValueError(
        "Inputs %s has unsupported rank."
        " Expected 2 or 4 but got %d" % (inputs.name, inputs_rank))
  # Bring 2-D inputs into 4-D format.
  if inputs_rank == 2:
    new_shape = [-1, 1, 1, num_channels]
    if data_format == "NCHW":
      new_shape = [-1, num_channels, 1, 1]
    inputs = tf.reshape(inputs, new_shape)

  # Execute a distributed batch normalization
  axis = 1 if data_format == "NCHW" else 3
  inputs = tf.cast(inputs, tf.float32)
  reduction_axes = [i for i in range(4) if i != axis]
  if use_cross_replica_mean:
    mean, variance = tpu_ops.cross_replica_moments(inputs, reduction_axes)
  else:
    counts, mean_ss, variance_ss, _ = tf.nn.sufficient_statistics(
        inputs, reduction_axes, keep_dims=False)
    mean, variance = tf.nn.normalize_moments(
        counts, mean_ss, variance_ss, shift=None)

  if use_moving_averages:
    mean, variance = _moving_moments_for_inference(
        mean=mean, variance=variance, is_training=is_training, decay=decay)
  else:
    mean, variance = _accumulated_moments_for_inference(
        mean=mean, variance=variance, is_training=is_training)

  outputs = tf.nn.batch_normalization(
      inputs,
      mean=mean,
      variance=variance,
      offset=None,
      scale=None,
      variance_epsilon=epsilon)
  outputs = tf.cast(outputs, inputs_dtype)

  # Bring 2-D inputs back into 2-D format.
  if inputs_rank == 2:
    outputs = tf.reshape(outputs, [-1] + inputs_shape[1:].as_list())
  outputs.set_shape(inputs_shape)
  return outputs


@gin.configurable(blacklist=["inputs"])
def no_batch_norm(inputs):
  return inputs


@gin.configurable(
    blacklist=["inputs", "is_training", "center", "scale", "name"])
def batch_norm(inputs, is_training, center=True, scale=True, name="batch_norm"):
  """Performs the vanilla batch normalization with trainable scaling and offset.

  Args:
    inputs: A tensor with 2 or 4 dimensions, where the first dimension is
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC`, and the second dimension if `data_format` is
      `NCHW`.
    is_training: Whether or not the layer is in training mode.
    center: If True, add offset of beta to normalized tensor.
    scale: If True, multiply by gamma. When the next layer is linear  this can
      be disabled since the scaling will be done by the next layer.
    name: Name of the variable scope.

  Returns:
    The normalized tensor with the same type and shape as `inputs`.
  """
  with tf.variable_scope(name, values=[inputs]):
    outputs = standardize_batch(inputs, is_training=is_training)
    num_channels = inputs.shape[-1].value

    # Allocate parameters for the trainable variables.
    collections = [tf.GraphKeys.MODEL_VARIABLES,
                   tf.GraphKeys.GLOBAL_VARIABLES]
    if scale:
      gamma = tf.get_variable(
          "gamma",
          [num_channels],
          collections=collections,
          initializer=tf.ones_initializer())
      outputs *= gamma
    if center:
      beta = tf.get_variable(
          "beta",
          [num_channels],
          collections=collections,
          initializer=tf.zeros_initializer())
      outputs += beta
    return outputs


@gin.configurable(whitelist=["num_hidden"])
def self_modulated_batch_norm(inputs, z, is_training, use_sn,
                              center=True, scale=True,
                              name="batch_norm", num_hidden=32):
  """Performs a self-modulated batch normalization.

  Details can be found in "On Self Modulation for Generative Adversarial
  Networks", Chen T. et al., 2018. [https://arxiv.org/abs/1810.01365]

  Like a normal batch normalization but the scale and offset are trainable
  transformation of `z`.

  Args:
    inputs: A tensor with 2 or 4 dimensions, where the first dimension is
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC`, and the second dimension if `data_format` is
      `NCHW`.
    z: 2-D tensor with shape [batch_size, ?] with the latent code.
    is_training: Whether or not the layer is in training mode.
    use_sn: Whether to apply spectral normalization to the weights of the
      hidden layer and the linear transformations.
    center: If True, add offset of beta to normalized tensor.
    scale: If True, multiply by gamma. When the next layer is linear  this can
      be disabled since the scaling will be done by the next layer.
    name: Name of the variable scope.
    num_hidden: Number of hidden units in the hidden layer. If 0 the scale and
      offset are simple linear transformations of `z`.

  Returns:
  """
  if z is None:
    raise ValueError("You must provide z for self modulation.")
  with tf.variable_scope(name, values=[inputs]):
    outputs = standardize_batch(inputs, is_training=is_training)
    num_channels = inputs.shape[-1].value

    with tf.variable_scope("sbn", values=[inputs, z]):
      h = z
      if num_hidden > 0:
        h = linear(h, num_hidden, scope="hidden", use_sn=use_sn)
        h = tf.nn.relu(h)
      if scale:
        gamma = linear(h, num_channels, scope="gamma", bias_start=1.0,
                       use_sn=use_sn)
        gamma = tf.reshape(gamma, [-1, 1, 1, num_channels])
        outputs *= gamma
      if center:
        beta = linear(h, num_channels, scope="beta", use_sn=use_sn)
        beta = tf.reshape(beta, [-1, 1, 1, num_channels])
        outputs += beta
      return outputs


@gin.configurable(whitelist=["use_bias"])
def conditional_batch_norm(inputs, y, is_training, use_sn, center=True,
                           scale=True, name="batch_norm", use_bias=False):
  """Conditional batch normalization."""
  if y is None:
    raise ValueError("You must provide y for conditional batch normalization.")
  if y.shape.ndims != 2:
    raise ValueError("Conditioning must have rank 2.")
  with tf.variable_scope(name, values=[inputs]):
    outputs = standardize_batch(inputs, is_training=is_training)
    num_channels = inputs.shape[-1].value
    with tf.variable_scope("condition", values=[inputs, y]):
      if scale:
        gamma = linear(y, num_channels, scope="gamma", use_sn=use_sn,
                       use_bias=use_bias)
        gamma = tf.reshape(gamma, [-1, 1, 1, num_channels])
        outputs *= gamma
      if center:
        beta = linear(y, num_channels, scope="beta", use_sn=use_sn,
                      use_bias=use_bias)
        beta = tf.reshape(beta, [-1, 1, 1, num_channels])
        outputs += beta
      return outputs


def layer_norm(input_, is_training, scope):
  return tf.contrib.layers.layer_norm(
      input_, trainable=is_training, scope=scope)


@gin.configurable(blacklist=["inputs"])
def spectral_norm(inputs, epsilon=1e-12, singular_value="left"):
  """Performs Spectral Normalization on a weight tensor.

  Details of why this is helpful for GAN's can be found in "Spectral
  Normalization for Generative Adversarial Networks", Miyato T. et al., 2018.
  [https://arxiv.org/abs/1802.05957].

  Args:
    inputs: The weight tensor to normalize.
    epsilon: Epsilon for L2 normalization.
    singular_value: Which first singular value to store (left or right). Use
      "auto" to automatically choose the one that has fewer dimensions.

  Returns:
    The normalized weight tensor.
  """
  if len(inputs.shape) < 2:
    raise ValueError(
        "Spectral norm can only be applied to multi-dimensional tensors")

  # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
  # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
  # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
  # layers that put output channels as last dimension. This implies that w
  # here is equivalent to w.T in the paper.
  w = tf.reshape(inputs, (-1, inputs.shape[-1]))

  # Choose whether to persist the first left or first right singular vector.
  # As the underlying matrix is PSD, this should be equivalent, but in practice
  # the shape of the persisted vector is different. Here one can choose whether
  # to maintain the left or right one, or pick the one which has the smaller
  # dimension. We use the same variable for the singular vector if we switch
  # from normal weights to EMA weights.
  var_name = inputs.name.replace("/ExponentialMovingAverage", "").split("/")[-1]
  var_name = var_name.split(":")[0] + "/u_var"
  if singular_value == "auto":
    singular_value = "left" if w.shape[0] <= w.shape[1] else "right"
  u_shape = (w.shape[0], 1) if singular_value == "left" else (1, w.shape[-1])
  u_var = tf.get_variable(
      var_name,
      shape=u_shape,
      dtype=w.dtype,
      initializer=tf.random_normal_initializer(),
      trainable=False)
  u = u_var

  # Use power iteration method to approximate the spectral norm.
  # The authors suggest that one round of power iteration was sufficient in the
  # actual experiment to achieve satisfactory performance.
  power_iteration_rounds = 1
  for _ in range(power_iteration_rounds):
    if singular_value == "left":
      # `v` approximates the first right singular vector of matrix `w`.
      v = tf.math.l2_normalize(
          tf.matmul(tf.transpose(w), u), axis=None, epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(w, v), axis=None, epsilon=epsilon)
    else:
      v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True),
                               epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(v, w), epsilon=epsilon)

  # Update the approximation.
  with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
    u = tf.identity(u)

  # The authors of SN-GAN chose to stop gradient propagating through u and v
  # and we maintain that option.
  u = tf.stop_gradient(u)
  v = tf.stop_gradient(v)

  if singular_value == "left":
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
  else:
    norm_value = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
  norm_value.shape.assert_is_fully_defined()
  norm_value.shape.assert_is_compatible_with([1, 1])

  w_normalized = w / norm_value

  # Deflate normalized weights to match the unnormalized tensor.
  w_tensor_normalized = tf.reshape(w_normalized, inputs.shape)
  return w_tensor_normalized


def linear(inputs, output_size, scope=None, stddev=0.02, bias_start=0.0,
           use_sn=False, use_bias=True):
  """Linear layer without the non-linear activation applied."""
  shape = inputs.get_shape().as_list()
  with tf.variable_scope(scope or "linear"):
    kernel = tf.get_variable(
        "kernel",
        [shape[1], output_size],
        initializer=weight_initializer(stddev=stddev))
    if use_sn:
      kernel = spectral_norm(kernel)
    outputs = tf.matmul(inputs, kernel)
    if use_bias:
      bias = tf.get_variable(
          "bias",
          [output_size],
          initializer=tf.constant_initializer(bias_start))
      outputs += bias
    return outputs


def conv2d(inputs, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           use_sn=False, use_bias=True):
  """Performs 2D convolution of the input."""
  with tf.variable_scope(name):
    w = tf.get_variable(
        "kernel", [k_h, k_w, inputs.shape[-1].value, output_dim],
        initializer=weight_initializer(stddev=stddev))
    if use_sn:
      w = spectral_norm(w)
    outputs = tf.nn.conv2d(inputs, w, strides=[1, d_h, d_w, 1], padding="SAME")
    if use_bias:
      bias = tf.get_variable(
          "bias", [output_dim], initializer=tf.constant_initializer(0.0))
      outputs += bias
  return outputs


conv1x1 = functools.partial(conv2d, k_h=1, k_w=1, d_h=1, d_w=1)


def deconv2d(inputs, output_shape, k_h, k_w, d_h, d_w,
             stddev=0.02, name="deconv2d", use_sn=False):
  """Performs transposed 2D convolution of the input."""
  with tf.variable_scope(name):
    w = tf.get_variable(
        "kernel", [k_h, k_w, output_shape[-1], inputs.get_shape()[-1]],
        initializer=weight_initializer(stddev=stddev))
    if use_sn:
      w = spectral_norm(w)
    deconv = tf.nn.conv2d_transpose(
        inputs, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    bias = tf.get_variable(
        "bias", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, bias), tf.shape(deconv))


def lrelu(inputs, leak=0.2, name="lrelu"):
  """Performs leaky-ReLU on the input."""
  return tf.maximum(inputs, leak * inputs, name=name)


def weight_norm_linear(input_, output_size,
                       init=False, init_scale=1.0,
                       name="wn_linear",
                       initializer=tf.truncated_normal_initializer,
                       stddev=0.02):
  """Linear layer with Weight Normalization (Salimans, Kingma '16)."""
  with tf.variable_scope(name):
    if init:
      v = tf.get_variable("V", [int(input_.get_shape()[1]), output_size],
                          tf.float32, initializer(0, stddev), trainable=True)
      v_norm = tf.nn.l2_normalize(v.initialized_value(), [0])
      x_init = tf.matmul(input_, v_norm)
      m_init, v_init = tf.nn.moments(x_init, [0])
      scale_init = init_scale / tf.sqrt(v_init + 1e-10)
      g = tf.get_variable("g", dtype=tf.float32,
                          initializer=scale_init, trainable=True)
      b = tf.get_variable("b", dtype=tf.float32, initializer=
                          -m_init*scale_init, trainable=True)
      x_init = tf.reshape(scale_init, [1, output_size]) * (
          x_init - tf.reshape(m_init, [1, output_size]))
      return x_init
    else:
      # Note that the original implementation uses Polyak averaging.
      v = tf.get_variable("V")
      g = tf.get_variable("g")
      b = tf.get_variable("b")
      tf.assert_variables_initialized([v, g, b])
      x = tf.matmul(input_, v)
      scaler = g / tf.sqrt(tf.reduce_sum(tf.square(v), [0]))
      x = tf.reshape(scaler, [1, output_size]) * x + tf.reshape(
          b, [1, output_size])
      return x


def weight_norm_conv2d(input_, output_dim,
                       k_h, k_w, d_h, d_w,
                       init, init_scale,
                       stddev=0.02,
                       name="wn_conv2d",
                       initializer=tf.truncated_normal_initializer):
  """Performs convolution with Weight Normalization."""
  with tf.variable_scope(name):
    if init:
      v = tf.get_variable(
          "V", [k_h, k_w] + [int(input_.get_shape()[-1]), output_dim],
          tf.float32, initializer(0, stddev), trainable=True)
      v_norm = tf.nn.l2_normalize(v.initialized_value(), [0, 1, 2])
      x_init = tf.nn.conv2d(input_, v_norm, strides=[1, d_h, d_w, 1],
                            padding="SAME")
      m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
      scale_init = init_scale / tf.sqrt(v_init + 1e-8)
      g = tf.get_variable(
          "g", dtype=tf.float32, initializer=scale_init, trainable=True)
      b = tf.get_variable(
          "b", dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
      x_init = tf.reshape(scale_init, [1, 1, 1, output_dim]) * (
          x_init - tf.reshape(m_init, [1, 1, 1, output_dim]))
      return x_init
    else:
      v = tf.get_variable("V")
      g = tf.get_variable("g")
      b = tf.get_variable("b")
      tf.assert_variables_initialized([v, g, b])
      w = tf.reshape(g, [1, 1, 1, output_dim]) * tf.nn.l2_normalize(
          v, [0, 1, 2])
      x = tf.nn.bias_add(
          tf.nn.conv2d(input_, w, [1, d_h, d_w, 1], padding="SAME"), b)
      return x


def weight_norm_deconv2d(x, output_dim,
                         k_h, k_w, d_h, d_w,
                         init=False, init_scale=1.0,
                         stddev=0.02,
                         name="wn_deconv2d",
                         initializer=tf.truncated_normal_initializer):
  """Performs Transposed Convolution with Weight Normalization."""
  xs = x.get_shape().as_list()
  target_shape = [xs[0], xs[1] * d_h, xs[2] * d_w, output_dim]
  with tf.variable_scope(name):
    if init:
      v = tf.get_variable(
          "V", [k_h, k_w] + [output_dim, int(x.get_shape()[-1])],
          tf.float32, initializer(0, stddev), trainable=True)
      v_norm = tf.nn.l2_normalize(v.initialized_value(), [0, 1, 3])
      x_init = tf.nn.conv2d_transpose(x, v_norm, target_shape,
                                      [1, d_h, d_w, 1], padding="SAME")
      m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
      scale_init = init_scale/tf.sqrt(v_init + 1e-8)
      g = tf.get_variable("g", dtype=tf.float32,
                          initializer=scale_init, trainable=True)
      b = tf.get_variable("b", dtype=tf.float32,
                          initializer=-m_init*scale_init, trainable=True)
      x_init = tf.reshape(scale_init, [1, 1, 1, output_dim]) * (
          x_init - tf.reshape(m_init, [1, 1, 1, output_dim]))
      return x_init
    else:
      v = tf.get_variable("v")
      g = tf.get_variable("g")
      b = tf.get_variable("b")
      tf.assert_variables_initialized([v, g, b])
      w = tf.reshape(g, [1, 1, output_dim, 1]) * tf.nn.l2_normalize(
          v, [0, 1, 3])
      x = tf.nn.conv2d_transpose(x, w, target_shape, strides=[1, d_h, d_w, 1],
                                 padding="SAME")
      x = tf.nn.bias_add(x, b)
      return x


def non_local_block(x, name, use_sn):
  """Self-attention (non-local) block.

  This method is used to exactly reproduce SAGAN and ignores Gin settings on
  weight initialization and spectral normalization.


  Args:
    x: Input tensor of shape [batch, h, w, c].
    name: Name of the variable scope.
    use_sn: Apply spectral norm to the weights.

  Returns:
    A tensor of the same shape after self-attention was applied.
  """
  def _spatial_flatten(inputs):
    shape = inputs.shape
    return tf.reshape(inputs, (-1, shape[1] * shape[2], shape[3]))

  with tf.variable_scope(name):
    h, w, num_channels = x.get_shape().as_list()[1:]
    num_channels_attn = num_channels // 8
    num_channels_g = num_channels // 2

    # Theta path
    theta = conv1x1(x, num_channels_attn, name="conv2d_theta", use_sn=use_sn,
                    use_bias=False)
    theta = _spatial_flatten(theta)

    # Phi path
    phi = conv1x1(x, num_channels_attn, name="conv2d_phi", use_sn=use_sn,
                  use_bias=False)
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    phi = _spatial_flatten(phi)

    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)

    # G path
    g = conv1x1(x, num_channels_g, name="conv2d_g", use_sn=use_sn,
                use_bias=False)
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = _spatial_flatten(g)

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [-1, h, w, num_channels_g])
    sigma = tf.get_variable("sigma", [], initializer=tf.zeros_initializer())
    attn_g = conv1x1(attn_g, num_channels, name="conv2d_attn_g", use_sn=use_sn,
                     use_bias=False)
    return x + sigma * attn_g
