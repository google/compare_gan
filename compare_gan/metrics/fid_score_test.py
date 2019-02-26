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

"""Tests for the FID score."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.metrics import fid_score as fid_score_lib

import numpy as np
import tensorflow as tf


class FIDScoreTest(tf.test.TestCase):

  def test_fid_computation(self):
    real_data = np.ones((100, 2))
    real_data[:50, 0] = 2
    gen_data = np.ones((100, 2)) * 9
    gen_data[50:, 0] = 2
    # mean(real_data) = [1.5, 1]
    # Cov(real_data) = [[ 0.2525, 0], [0, 0]]
    # mean(gen_data) = [5.5, 9]
    # Cov(gen_data) = [[12.37, 0], [0, 0]]
    result = fid_score_lib.compute_fid_from_activations(real_data, gen_data)
    self.assertNear(result, 89.091, 1e-4)

if __name__ == "__main__":
  tf.test.main()
