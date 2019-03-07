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

"""Tests for GANs with different regularizers."""

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
from compare_gan.gans.modular_gan import ModularGAN
import gin
import tensorflow as tf


FLAGS = flags.FLAGS
TEST_ARCHITECTURES = [c.RESNET5_ARCH, c.RESNET_BIGGAN_ARCH, c.RESNET_CIFAR_ARCH]
TEST_LOSSES = [loss_lib.non_saturating, loss_lib.wasserstein,
               loss_lib.least_squares, loss_lib.hinge]
TEST_PENALTIES = [penalty_lib.no_penalty, penalty_lib.dragan_penalty,
                  penalty_lib.wgangp_penalty, penalty_lib.l2_penalty]


class ModularGANConditionalTest(parameterized.TestCase,
                                test_utils.CompareGanTestCase):

  def _runSingleTrainingStep(self, architecture, loss_fn, penalty_fn,
                             labeled_dataset):
    parameters = {
        "architecture": architecture,
        "lambda": 1,
        "z_dim": 120,
    }
    with gin.unlock_config():
      gin.bind_parameter("penalty.fn", penalty_fn)
      gin.bind_parameter("loss.fn", loss_fn)
    model_dir = self._get_empty_model_dir()
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        conditional=True,
        model_dir=model_dir)
    estimator = gan.as_estimator(run_config, batch_size=2, use_tpu=False)
    estimator.train(gan.input_fn, steps=1)

  @parameterized.parameters(TEST_ARCHITECTURES)
  def testSingleTrainingStepArchitectures(self, architecture):
    self._runSingleTrainingStep(architecture, loss_lib.hinge,
                                penalty_lib.no_penalty, True)

  @parameterized.parameters(TEST_LOSSES)
  def testSingleTrainingStepLosses(self, loss_fn):
    self._runSingleTrainingStep(c.RESNET_CIFAR_ARCH, loss_fn,
                                penalty_lib.no_penalty, labeled_dataset=True)

  @parameterized.parameters(TEST_PENALTIES)
  def testSingleTrainingStepPenalties(self, penalty_fn):
    self._runSingleTrainingStep(c.RESNET_CIFAR_ARCH, loss_lib.hinge, penalty_fn,
                                labeled_dataset=True)

  def testUnlabledDatasetRaisesError(self):
    parameters = {
        "architecture": c.RESNET_CIFAR_ARCH,
        "lambda": 1,
        "z_dim": 120,
    }
    with gin.unlock_config():
      gin.bind_parameter("loss.fn", loss_lib.hinge)
    # Use dataset without labels.
    dataset = datasets.get_dataset("celeb_a")
    model_dir = self._get_empty_model_dir()
    with self.assertRaises(ValueError):
      gan = ModularGAN(
          dataset=dataset,
          parameters=parameters,
          conditional=True,
          model_dir=model_dir)
      del gan


if __name__ == "__main__":
  tf.test.main()
