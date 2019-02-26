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

"""Testing precision and recall computation on synthetic data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from compare_gan.metrics import prd_score as prd
import numpy as np


class PRDTest(unittest.TestCase):

  def test_compute_prd_no_overlap(self):
    eval_dist = [0, 1]
    ref_dist = [1, 0]
    result = np.ravel(prd.compute_prd(eval_dist, ref_dist))
    np.testing.assert_almost_equal(result, 0)

  def test_compute_prd_perfect_overlap(self):
    eval_dist = [1, 0]
    ref_dist = [1, 0]
    result = prd.compute_prd(eval_dist, ref_dist, num_angles=11)
    np.testing.assert_almost_equal([result[0][5], result[1][5]], [1, 1])

  def test_compute_prd_low_precision_high_recall(self):
    eval_dist = [0.5, 0.5]
    ref_dist = [1, 0]
    result = prd.compute_prd(eval_dist, ref_dist, num_angles=11)
    np.testing.assert_almost_equal(result[0][5], 0.5)
    np.testing.assert_almost_equal(result[1][5], 0.5)
    np.testing.assert_almost_equal(result[0][10], 0.5)
    np.testing.assert_almost_equal(result[1][1], 1)

  def test_compute_prd_high_precision_low_recall(self):
    eval_dist = [1, 0]
    ref_dist = [0.5, 0.5]
    result = prd.compute_prd(eval_dist, ref_dist, num_angles=11)
    np.testing.assert_almost_equal([result[0][5], result[1][5]], [0.5, 0.5])
    np.testing.assert_almost_equal(result[1][1], 0.5)
    np.testing.assert_almost_equal(result[0][10], 1)

  def test_compute_prd_bad_epsilon(self):
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], epsilon=0)
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], epsilon=1)
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], epsilon=-1)

  def test_compute_prd_bad_num_angles(self):
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], num_angles=0)
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], num_angles=1)
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], num_angles=-1)
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], num_angles=1e6+1)
    with self.assertRaises(ValueError):
      prd.compute_prd([1], [1], num_angles=2.5)

  def test__cluster_into_bins(self):
    eval_data = np.zeros([5, 4])
    ref_data = np.ones([5, 4])
    result = prd._cluster_into_bins(eval_data, ref_data, 3)

    self.assertEqual(len(result), 2)
    self.assertEqual(len(result[0]), 3)
    self.assertEqual(len(result[1]), 3)
    np.testing.assert_almost_equal(sum(result[0]), 1)
    np.testing.assert_almost_equal(sum(result[1]), 1)

  def test_compute_prd_from_embedding_mismatch_num_samples_should_fail(self):
    # Mismatch in number of samples with enforce_balance set to True
    with self.assertRaises(ValueError):
      prd.compute_prd_from_embedding(
          np.array([[0], [0], [1]]), np.array([[0], [1]]), num_clusters=2,
          enforce_balance=True)

  def test_compute_prd_from_embedding_mismatch_num_samples_should_work(self):
    # Mismatch in number of samples with enforce_balance set to False
    try:
      prd.compute_prd_from_embedding(
          np.array([[0], [0], [1]]), np.array([[0], [1]]), num_clusters=2,
          enforce_balance=False)
    except ValueError:
      self.fail(
          'compute_prd_from_embedding should not raise a ValueError when '
          'enforce_balance is set to False.')

  def test__prd_to_f_beta_correct_computation(self):
    precision = np.array([1, 1, 0, 0, 0.5, 1, 0.5])
    recall = np.array([1, 0, 1, 0, 0.5, 0.5, 1])
    expected = np.array([1, 0, 0, 0, 0.5, 2/3, 2/3])
    with np.errstate(invalid='ignore'):
      result = prd._prd_to_f_beta(precision, recall, beta=1)
    np.testing.assert_almost_equal(result, expected)

    expected = np.array([1, 0, 0, 0, 0.5, 5/9, 5/6])
    with np.errstate(invalid='ignore'):
      result = prd._prd_to_f_beta(precision, recall, beta=2)
    np.testing.assert_almost_equal(result, expected)

    expected = np.array([1, 0, 0, 0, 0.5, 5/6, 5/9])
    with np.errstate(invalid='ignore'):
      result = prd._prd_to_f_beta(precision, recall, beta=1/2)
    np.testing.assert_almost_equal(result, expected)

    result = prd._prd_to_f_beta(np.array([]), np.array([]), beta=1)
    expected = np.array([])
    np.testing.assert_almost_equal(result, expected)

  def test__prd_to_f_beta_bad_beta(self):
    with self.assertRaises(ValueError):
      prd._prd_to_f_beta(np.ones(1), np.ones(1), beta=0)
    with self.assertRaises(ValueError):
      prd._prd_to_f_beta(np.ones(1), np.ones(1), beta=-3)

  def test__prd_to_f_beta_bad_precision_or_recall(self):
    with self.assertRaises(ValueError):
      prd._prd_to_f_beta(-np.ones(1), np.ones(1), beta=1)
    with self.assertRaises(ValueError):
      prd._prd_to_f_beta(np.ones(1), -np.ones(1), beta=1)

  def test_plot_not_enough_labels(self):
    with self.assertRaises(ValueError):
      prd.plot(np.zeros([3, 2, 5]), labels=['1', '2'])

  def test_plot_too_many_labels(self):
    with self.assertRaises(ValueError):
      prd.plot(np.zeros([1, 2, 5]), labels=['1', '2', '3'])


if __name__ == '__main__':
  unittest.main()
