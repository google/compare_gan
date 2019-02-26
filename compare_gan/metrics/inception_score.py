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

"""Implementation of the Inception Score.

Implemented as a wrapper around the tensorflow_gan library. The details can be
found in "Improved Techniques for Training GANs", Salimans et al.
[https://arxiv.org/abs/1606.03498].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from compare_gan.metrics import eval_task
import tensorflow as tf
import tensorflow_gan as tfgan


class InceptionScoreTask(eval_task.EvalTask):
  """Task that computes inception score for the generated images."""

  _LABEL = "inception_score"

  def run_after_session(self, fake_dset, real_dest):
    del real_dest
    logging.info("Computing inception score.")
    with tf.Graph().as_default():
      fake_logits = tf.convert_to_tensor(fake_dset.logits)
      inception_score = tfgan.eval.classifier_score_from_logits(fake_logits)
      with self._create_session() as sess:
        inception_score = sess.run(inception_score)
      logging.info("Inception score: %.3f", inception_score)
    return {self._LABEL: inception_score}
