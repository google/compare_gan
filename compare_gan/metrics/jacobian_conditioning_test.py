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

"""Tests for Jacobian Conditioning metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.metrics import jacobian_conditioning
import mock
import numpy as np
from six.moves import range
import tensorflow as tf


_BATCH_SIZE = 32


def SlowJacobian(xs, fx):
  """Computes df/dx matrix.

  As jacobian_conditioning.compute_jacobian, but explicitly loops over
  dimensions of f.

  Args:
    xs: input tensor(s) of arbitrary shape.
    fx: f(x) tensor of arbitrary shape.

  Returns:
    df/dx tensor.
  """
  fxs = tf.unstack(fx, axis=-1)
  grads = [tf.gradients(fx_i, xs) for fx_i in fxs]
  grads = [grad[0] for grad in grads]
  df_dx = tf.stack(grads, axis=1)
  return df_dx


class JacobianConditioningTest(tf.test.TestCase):

  def test_jacobian_simple_case(self):
    x = tf.random_normal([_BATCH_SIZE, 2])
    W = tf.constant([[2., -1.], [1.5, 1.]])  # pylint: disable=invalid-name
    f = tf.matmul(x, W)
    j_tensor = jacobian_conditioning.compute_jacobian(xs=x, fx=f)
    with tf.Session() as sess:
      jacobian = sess.run(j_tensor)

    # Transpose of W in 'expected' is expected because in vector notation
    # f = W^T * x.
    expected = tf.tile([[[2, 1.5], [-1, 1]]], [_BATCH_SIZE, 1, 1])
    self.assertAllClose(jacobian, expected)

  def test_jacobian_against_slow_version(self):
    x = tf.random_normal([_BATCH_SIZE, 2])
    h1 = tf.contrib.layers.fully_connected(x, 20)
    h2 = tf.contrib.layers.fully_connected(h1, 20)
    f = tf.contrib.layers.fully_connected(h2, 10)

    j_slow_tensor = SlowJacobian(xs=x, fx=f)
    j_fast_tensor = jacobian_conditioning.compute_jacobian(xs=x, fx=f)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      j_fast, j_slow = sess.run([j_fast_tensor, j_slow_tensor])
    self.assertAllClose(j_fast, j_slow)

  def test_jacobian_numerically(self):
    x = tf.random_normal([_BATCH_SIZE, 2])
    h1 = tf.contrib.layers.fully_connected(x, 20)
    h2 = tf.contrib.layers.fully_connected(h1, 20)
    f = tf.contrib.layers.fully_connected(h2, 10)
    j_tensor = jacobian_conditioning.compute_jacobian(xs=x, fx=f)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      x_np = sess.run(x)
      jacobian = sess.run(j_tensor, feed_dict={x: x_np})

      # Test 10 random elements.
      for _ in range(10):
        # Pick a random element of Jacobian to test.
        batch_idx = np.random.randint(_BATCH_SIZE)
        x_idx = np.random.randint(2)
        f_idx = np.random.randint(10)

        # Test with finite differences.
        epsilon = 1e-4

        x_plus = x_np.copy()
        x_plus[batch_idx, x_idx] += epsilon
        f_plus = sess.run(f, feed_dict={x: x_plus})[batch_idx, f_idx]

        x_minus = x_np.copy()
        x_minus[batch_idx, x_idx] -= epsilon
        f_minus = sess.run(f, feed_dict={x: x_minus})[batch_idx, f_idx]

        self.assertAllClose(
            jacobian[batch_idx, f_idx, x_idx],
            (f_plus - f_minus) / (2. * epsilon),
            rtol=1e-3,
            atol=1e-3)

  def test_analyze_metric_tensor(self):
    # Assumes NumPy works, just tests that output shapes are as expected.
    jacobian = np.random.normal(0, 1, (_BATCH_SIZE, 2, 10))
    metric_tensor = np.matmul(np.transpose(jacobian, [0, 2, 1]), jacobian)
    result_dict = jacobian_conditioning._analyze_metric_tensor(metric_tensor)
    self.assertAllEqual(result_dict['eigenvalues'].shape, [_BATCH_SIZE, 10])
    self.assertAllEqual(result_dict['logdet'].shape, [_BATCH_SIZE])
    self.assertAllEqual(result_dict['log_condition_number'].shape,
                        [_BATCH_SIZE])

  def test_analyze_jacobian(self):
    m = mock.patch.object(
        jacobian_conditioning, '_analyze_metric_tensor', new=lambda x: x)
    m.start()
    jacobian = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
    result_dict = jacobian_conditioning.analyze_jacobian(jacobian)
    self.assertAllEqual(result_dict['metric_tensor'],
                        [[[10, 14], [14, 20]], [[40, 56], [56, 80]]])
    self.assertAllEqual(result_dict['mean_metric_tensor'],
                        [[[25, 35], [35, 50]]])
    m.stop()


if __name__ == '__main__':
  tf.test.main()
