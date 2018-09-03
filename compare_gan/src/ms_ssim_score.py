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

"""MS-SSIM metrics for image diversity evaluation.

More details could be found from section 5.3:
https://arxiv.org/pdf/1710.08446.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from compare_gan.src import image_similarity

import numpy as np
from six.moves import range
import tensorflow as tf

logging = tf.logging


def get_metric_function(generated_images, num_batches):
  """Get a fn returning the ms ssim score for generated images.

  Args:
    generated_images: TF Tensor of shape [batch_size, dim, dim, 3] which
      evaluates to a batch of generated images. Should be in range [0..255].
    num_eval_images: Number of (generated/ground_truth) images to evaluate.

  Returns:
    eval_fn: a function which takes a session as an argument and returns the
      average ms ssim score among all the possible image pairs from
      generated_images.
  """
  batch_size = int(generated_images.get_shape()[0])
  assert batch_size > 1

  # Generate all possible image pairs from input set of imgs.
  pair1 = tf.tile(generated_images, [batch_size, 1, 1, 1])
  pair2 = tf.reshape(
      tf.tile(generated_images, [1, batch_size, 1, 1]), [
          batch_size * batch_size, generated_images.shape[1],
          generated_images.shape[2], generated_images.shape[3]
      ])

  # Compute the mean of the scores (but ignore the 'identical' images - which
  # should get 1.0 from the MultiscaleSSIM)
  score = tf.reduce_sum(image_similarity.MultiscaleSSIM(pair1,
                                                        pair2)) - batch_size
  score = tf.div(score, batch_size * batch_size - batch_size)

  # Define a function which wraps some session.run calls to generate a large
  # number of images and compute multiscale ssim metric on them.
  def eval_fn(session):
    """Function which wraps session.run calls to compute given metric."""
    logging.info("Computing MS-SSIM score...")
    scores = []
    for _ in range(num_batches):
      scores.append(session.run(score))

    result = np.mean(scores)
    return result

  return eval_fn
