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

"""Utilities library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tensorflow as tf


def check_folder(log_dir):
  if not tf.gfile.IsDirectory(log_dir):
    tf.gfile.MakeDirs(log_dir)
  return log_dir


def save_images(images, image_path):
  with tf.gfile.Open(image_path, "wb") as f:
    scipy.misc.imsave(f, images * 255.0)


def rotate_images(images, rot90_scalars=(0, 1, 2, 3)):
  """Return the input image and its 90, 180, and 270 degree rotations."""
  images_rotated = [
      images,  # 0 degree
      tf.image.flip_up_down(tf.image.transpose_image(images)),  # 90 degrees
      tf.image.flip_left_right(tf.image.flip_up_down(images)),  # 180 degrees
      tf.image.transpose_image(tf.image.flip_up_down(images))  # 270 degrees
  ]

  results = tf.stack([images_rotated[i] for i in rot90_scalars])
  results = tf.reshape(results,
                       [-1] + images.get_shape().as_list()[1:])
  return results


def gaussian(batch_size, n_dim, mean=0., var=1.):
  return np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
