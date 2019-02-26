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

"""Implementation of the Jacobian Conditioning metrics.

The details can be found in "Is Generator Conditioning Causally Related to
GAN Performance?", Odena et al. [https://arxiv.org/abs/1802.08768].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.metrics import eval_task

import numpy as np
import tensorflow as tf


class GeneratorConditionNumberTask(eval_task.EvalTask):
  """Computes the generator condition number.

  Computes the condition number for metric Tensor of the generator Jacobian.
  This condition number is computed locally for each z sample in a minibatch.
  Returns the mean log condition number and standard deviation across the
  minibatch.

  Follows the methods in https://arxiv.org/abs/1802.08768.
  """

  _CONDITION_NUMBER_COUNT = "log_condition_number_count"
  _CONDITION_NUMBER_MEAN = "log_condition_number_mean"
  _CONDITION_NUMBER_STD = "log_condition_number_std"

  def metric_list(self):
    return frozenset([
        self._CONDITION_NUMBER_COUNT, self._CONDITION_NUMBER_MEAN,
        self._CONDITION_NUMBER_STD
    ])

  def run_in_session(self, options, sess, gan, real_images):
    del options, real_images
    result_dict = {}
    result = compute_generator_condition_number(sess, gan)
    result_dict[self._CONDITION_NUMBER_COUNT] = len(result)
    result_dict[self._CONDITION_NUMBER_MEAN] = np.mean(result)
    result_dict[self._CONDITION_NUMBER_STD] = np.std(result)
    return result_dict


def compute_generator_condition_number(sess, gan):
  """Computes the generator condition number.

  Computes the Jacobian of the generator in session, then postprocesses to get
  the condition number.

  Args:
    sess: tf.Session object.
    gan: AbstractGAN object, that is already present in the current tf.Graph.

  Returns:
    A list of length gan.batch_size. Each element is the condition number
    computed at a single z sample within a minibatch.
  """
  shape = gan.fake_images.get_shape().as_list()
  flat_generator_output = tf.reshape(
      gan.fake_images, [gan.batch_size, np.prod(shape[1:])])
  tf_jacobian = compute_jacobian(
      xs=gan.z, fx=flat_generator_output)
  z_sample = gan.z_generator(gan.batch_size, gan.z_dim)
  np_jacobian = sess.run(tf_jacobian, feed_dict={gan.z: z_sample})
  result_dict = analyze_jacobian(np_jacobian)
  return result_dict["metric_tensor"]["log_condition_number"]


def compute_jacobian(xs, fx):
  """Computes df/dx matrix.

  We assume x and fx are both batched, so the shape of the Jacobian is:
  [fx.shape[0]] + fx.shape[1:] + xs.shape[1:]

  This function computes the grads inside a TF loop so that we don't
  end up storing many extra copies of the function we are taking the
  Jacobian of.

  Args:
    xs: input tensor(s) of arbitrary shape.
    fx: f(x) tensor of arbitrary shape.

  Returns:
    df/dx tensor of shape [fx.shape[0], fx.shape[1], xs.shape[1]].
  """
  # Declares an iterator and tensor array loop variables for the gradients.
  n = fx.get_shape().as_list()[1]
  loop_vars = [tf.constant(0, tf.int32), tf.TensorArray(xs.dtype, n)]

  def accumulator(j, result):
    return (j + 1, result.write(j, tf.gradients(fx[:, j], xs)[0]))

  # Iterates over all elements of the gradient and computes all partial
  # derivatives.
  _, df_dxs = tf.while_loop(lambda j, _: j < n, accumulator, loop_vars)

  df_dx = df_dxs.stack()
  df_dx = tf.transpose(df_dx, perm=[1, 0, 2])

  return df_dx


def _analyze_metric_tensor(metric_tensor):
  """Analyzes a metric tensor.

  Args:
    metric_tensor: A numpy array of shape [batch, dim, dim]

  Returns:
    A dict containing spectral statstics.
  """
  # eigenvalues will have shape [batch, dim].
  eigenvalues, _ = np.linalg.eig(metric_tensor)

  # Shape [batch,].
  condition_number = np.linalg.cond(metric_tensor)
  log_condition_number = np.log(condition_number)
  (_, logdet) = np.linalg.slogdet(metric_tensor)

  return {
      "eigenvalues": eigenvalues,
      "logdet": logdet,
      "log_condition_number": log_condition_number
  }


def analyze_jacobian(jacobian_array):
  """Computes eigenvalue statistics of the Jacobian.

  Computes the eigenvalues and condition number of the metric tensor for the
  Jacobian evaluated at each element of the batch and the mean metric tensor
  across the batch.

  Args:
    jacobian_array: A numpy array holding the Jacobian.

  Returns:
    A dict of spectral statistics with two elements, one containing stats
    for every metric tensor in the batch, another for the mean metric tensor.
  """
  # Shape [batch, x_dim, fx_dim].
  jacobian_transpose = np.transpose(jacobian_array, [0, 2, 1])

  # Shape [batch, x_dim, x_dim].
  metric_tensor = np.matmul(jacobian_transpose, jacobian_array)

  mean_metric_tensor = np.mean(metric_tensor, 0)
  # Reshapes to have a dummy batch dimension.
  mean_metric_tensor = np.reshape(mean_metric_tensor,
                                  (1,) + metric_tensor.shape[1:])
  return {
      "metric_tensor": _analyze_metric_tensor(metric_tensor),
      "mean_metric_tensor": _analyze_metric_tensor(mean_metric_tensor)
  }
