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

import os

from absl import flags
from absl.testing import parameterized
from compare_gan import datasets
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans import penalty_lib
from compare_gan.gans.modular_gan import ModularGAN
import gin
import tensorflow as tf


FLAGS = flags.FLAGS
TEST_ARCHITECTURES = [c.RESNET_CIFAR, c.RESNET5_ARCH, c.RESNET5_BIGGAN_ARCH]
TEST_LOSSES = [loss_lib.non_saturating, loss_lib.wasserstein,
               loss_lib.least_squares, loss_lib.hinge]
TEST_PENALTIES = [penalty_lib.no_penalty, penalty_lib.dragan_penalty,
                  penalty_lib.wgangp_penalty, penalty_lib.l2_penalty]


class ModularGANConditionalTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ModularGANConditionalTest, self).setUp()
    FLAGS.data_fake_dataset = True
    self.model_dir = os.path.join(FLAGS.test_tmpdir, "model_dir")
    if tf.gfile.Exists(self.model_dir):
      tf.gfile.DeleteRecursively(self.model_dir)

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
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=self.model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        conditional=True,
        model_dir=self.model_dir)
    estimator = gan.as_estimator(run_config, batch_size=2, use_tpu=False)
    estimator.train(gan.input_fn, steps=1)

  @parameterized.parameters(TEST_ARCHITECTURES)
  def testSingleTrainingStepArchitectures(self, architecture):
    self._runSingleTrainingStep(architecture, loss_lib.hinge,
                                penalty_lib.no_penalty, True)

  @parameterized.parameters(TEST_LOSSES)
  def testSingleTrainingStepLosses(self, loss_fn):
    self._runSingleTrainingStep(c.RESNET_CIFAR, loss_fn, penalty_lib.no_penalty,
                                True)

  @parameterized.parameters(TEST_PENALTIES)
  def testSingleTrainingStepPenalties(self, penalty_fn):
    self._runSingleTrainingStep(c.RESNET_CIFAR, loss_lib.hinge, penalty_fn,
                                True)

  def testUnlabledDatasetRaisesError(self):
    parameters = {
        "architecture": c.RESNET_CIFAR,
        "lambda": 1,
        "z_dim": 120,
    }
    with gin.unlock_config():
      gin.bind_parameter("loss.fn", loss_lib.hinge)
    # Use dataset without labels.
    dataset = datasets.get_dataset("celeb_a")
    with self.assertRaises(ValueError):
      gan = ModularGAN(
          dataset=dataset,
          parameters=parameters,
          conditional=True,
          model_dir=self.model_dir)
      del gan


if __name__ == "__main__":
  tf.test.main()
