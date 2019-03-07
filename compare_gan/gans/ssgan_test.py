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

"""Tests for SSGANs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
from compare_gan import datasets
from compare_gan import test_utils
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans import penalty_lib
from compare_gan.gans.ssgan import SSGAN
import gin
import tensorflow as tf

FLAGS = flags.FLAGS
TEST_ARCHITECTURES = [c.RESNET_CIFAR_ARCH, c.SNDCGAN_ARCH, c.RESNET5_ARCH]
TEST_LOSSES = [loss_lib.non_saturating, loss_lib.hinge]
TEST_PENALTIES = [penalty_lib.no_penalty, penalty_lib.wgangp_penalty]


class SSGANTest(parameterized.TestCase, test_utils.CompareGanTestCase):

  def _runSingleTrainingStep(self, architecture, loss_fn, penalty_fn):
    parameters = {
        "architecture": architecture,
        "lambda": 1,
        "z_dim": 128,
    }
    with gin.unlock_config():
      gin.bind_parameter("penalty.fn", penalty_fn)
      gin.bind_parameter("loss.fn", loss_fn)
    model_dir = self._get_empty_model_dir()
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    dataset = datasets.get_dataset("cifar10")
    gan = SSGAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=model_dir,
        g_optimizer_fn=tf.train.AdamOptimizer,
        g_lr=0.0002,
        rotated_batch_size=4)
    estimator = gan.as_estimator(run_config, batch_size=2, use_tpu=False)
    estimator.train(gan.input_fn, steps=1)

  @parameterized.parameters(TEST_ARCHITECTURES)
  def testSingleTrainingStepArchitectures(self, architecture):
    self._runSingleTrainingStep(architecture, loss_lib.hinge,
                                penalty_lib.no_penalty)

  @parameterized.parameters(TEST_LOSSES)
  def testSingleTrainingStepLosses(self, loss_fn):
    self._runSingleTrainingStep(c.RESNET_CIFAR_ARCH, loss_fn,
                                penalty_lib.no_penalty)

  @parameterized.parameters(TEST_PENALTIES)
  def testSingleTrainingStepPenalties(self, penalty_fn):
    self._runSingleTrainingStep(c.RESNET_CIFAR_ARCH, loss_lib.hinge, penalty_fn)


if __name__ == "__main__":
  tf.test.main()
