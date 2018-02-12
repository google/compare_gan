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

"""Tests for compare_gan.params."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src import params
import tensorflow as tf


class ParamsTest(tf.test.TestCase):

  def testParameterRanges(self):
    training_parameters = params.GetParameters(
        "WGAN", "mnist", "wide")
    self.assertEqual(len(training_parameters.keys()), 5)

    training_parameters = params.GetParameters(
        "BEGAN", "mnist", "wide")
    self.assertEqual(len(training_parameters.keys()), 6)


if __name__ == "__main__":
  tf.test.main()
