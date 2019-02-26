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

"""Tests for the fractal dimension metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.metrics import fractal_dimension as fractal_dimension_lib

import numpy as np
import tensorflow as tf


class FractalDimensionTest(tf.test.TestCase):

  def test_straight_line(self):
    """The fractal dimension of a 1D line must lie near 1.0."""
    self.assertAllClose(
        fractal_dimension_lib.compute_fractal_dimension(
            np.random.uniform(size=(10000, 1))), 1.0, atol=0.05)

  def test_square(self):
    """The fractal dimension of a 2D square must lie near 2.0."""
    self.assertAllClose(
        fractal_dimension_lib.compute_fractal_dimension(
            np.random.uniform(size=(10000, 2))), 2.0, atol=0.1)
