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

"""Implementation of the Frechet Inception Distance.

Implemented as a wrapper around the tf.contrib.gan library. The details can be
found in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
Equilibrium", Heusel et al. [https://arxiv.org/abs/1706.08500].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from compare_gan.metrics import eval_task

import tensorflow as tf
import tensorflow_gan as tfgan


# Special value returned when FID code returned exception.
FID_CODE_FAILED = 4242.0


class FIDScoreTask(eval_task.EvalTask):
  """Evaluation task for the FID score."""

  _LABEL = "fid_score"

  def run_after_session(self, fake_dset, real_dset):
    logging.info("Calculating FID.")
    with tf.Graph().as_default():
      fake_activations = tf.convert_to_tensor(fake_dset.activations)
      real_activations = tf.convert_to_tensor(real_dset.activations)
      fid = tfgan.eval.frechet_classifier_distance_from_activations(
          real_activations=real_activations,
          generated_activations=fake_activations)
      with self._create_session() as sess:
        fid = sess.run(fid)
      logging.info("Frechet Inception Distance: %.3f.", fid)
      return {self._LABEL: fid}


def compute_fid_from_activations(fake_activations, real_activations):
  """Returns the FID based on activations.

  Args:
    fake_activations: NumPy array with fake activations.
    real_activations: NumPy array with real activations.
  Returns:
    A float, the Frechet Inception Distance.
  """
  logging.info("Computing FID score.")
  assert fake_activations.shape == real_activations.shape
  with tf.Session(graph=tf.Graph()) as sess:
    fake_activations = tf.convert_to_tensor(fake_activations)
    real_activations = tf.convert_to_tensor(real_activations)
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        real_activations=real_activations,
        generated_activations=fake_activations)
    return sess.run(fid)
