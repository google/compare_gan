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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
from six.moves import map
from six.moves import range
import tensorflow as tf


def check_folder(log_dir):
  if not tf.gfile.IsDirectory(log_dir):
    tf.gfile.MakeDirs(log_dir)
  return log_dir


def save_images(images, image_path):
  with tf.gfile.Open(image_path, "wb") as f:
    scipy.misc.imsave(f, images * 255.0)


def gaussian(batch_size, n_dim, mean=0., var=1.):
  return np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def batch_norm(input_, is_training, scope):
  return tf.contrib.layers.batch_norm(
      input_,
      decay=0.999,
      epsilon=0.001,
      updates_collections=None,
      scale=True,
      fused=False,
      is_training=is_training,
      scope=scope)


def layer_norm(input_, is_training, scope):
  return tf.contrib.layers.layer_norm(
      input_, trainable=is_training, scope=scope)


def spectral_norm(input_):
  """Performs Spectral Normalization on a weight tensor."""
  if len(input_.shape) < 2:
    raise ValueError(
        "Spectral norm can only be applied to multi-dimensional tensors")

  # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
  # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
  # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
  # (KH * KW * C_in, C_out), and similarly for other layers that put output
  # channels as last dimension.
  # n.b. this means that w here is equivalent to w.T in the paper.
  w = tf.reshape(input_, (-1, input_.shape[-1]))

  # Persisted approximation of first left singular vector of matrix `w`.

  u_var = tf.get_variable(
      input_.name.replace(":", "") + "/u_var",
      shape=(w.shape[0], 1),
      dtype=w.dtype,
      initializer=tf.random_normal_initializer(),
      trainable=False)
  u = u_var

  # Use power iteration method to approximate spectral norm.
  # The authors suggest that "one round of power iteration was sufficient in the
  # actual experiment to achieve satisfactory performance". According to
  # observation, the spectral norm become very accurate after ~20 steps.

  power_iteration_rounds = 1
  for _ in range(power_iteration_rounds):
    # `v` approximates the first right singular vector of matrix `w`.
    v = tf.nn.l2_normalize(
        tf.matmul(tf.transpose(w), u), axis=None, epsilon=1e-12)
    u = tf.nn.l2_normalize(tf.matmul(w, v), axis=None, epsilon=1e-12)

  # Update persisted approximation.
  with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
    u = tf.identity(u)

  # The authors of SN-GAN chose to stop gradient propagating through u and v.
  # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
  # seem to hinder either so it's kept in order to be a faithful implementation.
  u = tf.stop_gradient(u)
  v = tf.stop_gradient(v)

  # Largest singular value of `w`.
  norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
  norm_value.shape.assert_is_fully_defined()
  norm_value.shape.assert_is_compatible_with([1, 1])

  w_normalized = w / norm_value

  # Unflatten normalized weights to match the unnormalized tensor.
  w_tensor_normalized = tf.reshape(w_normalized, input_.shape)
  return w_tensor_normalized


def linear(input_,
           output_size,
           scope=None,
           stddev=0.02,
           bias_start=0.0,
           use_sn=False):

  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", [shape[1], output_size],
        tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable(
        "bias", [output_size], initializer=tf.constant_initializer(bias_start))
    if use_sn:
      return tf.matmul(input_, spectral_norm(matrix)) + bias
    else:
      return tf.matmul(input_, matrix) + bias


def spectral_norm_update_ops(var_list, weight_getter):
  update_ops = []
  print(" [*] Spectral norm layers")
  layer = 0
  for var in var_list:
    if weight_getter.match(var.name):
      layer += 1
      print("     %d. %s" % (layer, var.name))
      # Alternative solution here is keep spectral norm and original weight
      # matrix separately, and only normalize the weight matrix if needed.
      # But as spectral norm converges to 1.0 very quickly, it should be very
      # minor accuracy diff caused by float value division.
      update_ops.append(tf.assign(var, spectral_norm(var)))
  return update_ops


def spectral_norm_svd(input_):
  if len(input_.shape) < 2:
    raise ValueError(
        "Spectral norm can only be applied to multi-dimensional tensors")

  w = tf.reshape(input_, (-1, input_.shape[-1]))
  s, _, _ = tf.svd(w)
  return s[0]


def spectral_norm_value(var_list, weight_getter):
  """Compute spectral norm value using svd, for debug purpose."""
  norms = {}
  for var in var_list:
    if weight_getter.match(var.name):
      norms[var.name] = spectral_norm_svd(var)
  return norms


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d",
           initializer=tf.truncated_normal_initializer, use_sn=False):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=initializer(stddev=stddev))
    if use_sn:
      conv = tf.nn.conv2d(
          input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding="SAME")
    else:
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
    biases = tf.get_variable(
        "biases", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w,
             stddev=0.02, name="deconv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(
        input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    biases = tf.get_variable(
        "biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


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
  """Convolution with Weight Normalization (Salimans, Kingma '16)."""
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
  """Transposed Convolution with Weight Normalization (Salimans, Kingma '16)."""
  xs = list(map(int, x.get_shape()))
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
