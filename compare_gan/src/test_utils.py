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

"""Helper functions for test case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def load_fake_dataset(options, num_examples=100):
  """Returns a fake dataset with with fake images and labels.

  Args:
    options: Dictionary with options. Must contain "input_height",
        "input_width" and "c_dim".
    num_examples: Number of examples to generate. You can always call
        repeat() and the dataset to iterate for longer.
  Returns:
    `tf.data.Dataset` with `num_examples`. Each example is a tuple of an
    image and a label. The image is a 3D tensor with value in [0, 1]. The label
    is always 0.
  """
  image_shape = (num_examples, options["input_height"],
                 options["input_width"], options["c_dim"])
  return tf.data.Dataset.from_tensor_slices(
      (np.random.uniform(size=image_shape), np.zeros(num_examples)))
