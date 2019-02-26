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

"""Implementation of the MS-SSIM metric.

The details on the application of this metric to GANs can be found in
Section 5.3 of "Many Paths to Equilibrium: GANs Do Not Need to Decrease a
Divergence At Every Step", Fedus*, Rosca* et al.
[https://arxiv.org/abs/1710.08446].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from compare_gan.metrics import eval_task
from compare_gan.metrics import image_similarity

import numpy as np
from six.moves import range
import tensorflow as tf


class MultiscaleSSIMTask(eval_task.EvalTask):
  """Task that computes MSSIMScore for generated images."""

  _LABEL = "ms_ssim"

  def run_after_session(self, options, eval_data_fake, eval_data_real=None):
    del options, eval_data_real
    score = _compute_multiscale_ssim_score(eval_data_fake.images)
    return {self._LABEL: score}


def _compute_multiscale_ssim_score(fake_images):
  """Compute ms-ssim score ."""
  batch_size = 64
  with tf.Graph().as_default():
    fake_images_batch = tf.train.shuffle_batch(
        [tf.convert_to_tensor(fake_images, dtype=tf.float32)],
        capacity=16*batch_size,
        min_after_dequeue=8*batch_size,
        num_threads=4,
        enqueue_many=True,
        batch_size=batch_size)

    # Following section 5.3 of https://arxiv.org/pdf/1710.08446.pdf, we only
    # evaluate 5 batches of the generated images.
    eval_fn = compute_msssim(
        generated_images=fake_images_batch, num_batches=5)
    with tf.train.MonitoredTrainingSession() as sess:
      score = eval_fn(sess)
  return score


def compute_msssim(generated_images, num_batches):
  """Get a fn returning the ms ssim score for generated images.

  Args:
    generated_images: TF Tensor of shape [batch_size, dim, dim, 3] which
      evaluates to a batch of generated images. Should be in range [0..255].
    num_batches: Number of batches to consider.

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
  score = tf.reduce_sum(image_similarity.multiscale_ssim(pair1, pair2))
  score -= batch_size
  score = tf.div(score, batch_size * batch_size - batch_size)

  # Define a function which wraps some session.run calls to generate a large
  # number of images and compute multiscale ssim metric on them.
  def _eval_fn(session):
    """Function which wraps session.run calls to compute given metric."""
    logging.info("Computing MS-SSIM score...")
    scores = []
    for _ in range(num_batches):
      scores.append(session.run(score))

    result = np.mean(scores)
    return result
  return _eval_fn
