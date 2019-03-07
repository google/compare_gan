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

"""Test various (unconditional) configurations of ModularGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

from absl import flags
from absl.testing import parameterized
from compare_gan import datasets
from compare_gan import test_utils
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans import penalty_lib
from compare_gan.gans.modular_gan import ModularGAN
import gin
import numpy as np
from six.moves import range
import tensorflow as tf


FLAGS = flags.FLAGS
TEST_ARCHITECTURES = [c.INFOGAN_ARCH, c.DCGAN_ARCH, c.RESNET_CIFAR_ARCH,
                      c.SNDCGAN_ARCH, c.RESNET5_ARCH]
TEST_LOSSES = [loss_lib.non_saturating, loss_lib.wasserstein,
               loss_lib.least_squares, loss_lib.hinge]
TEST_PENALTIES = [penalty_lib.no_penalty, penalty_lib.dragan_penalty,
                  penalty_lib.wgangp_penalty, penalty_lib.l2_penalty]
GENERATOR_TRAINED_IN_STEPS = [
    # disc_iters=1.
    [True, True, True],
    # disc_iters=2.
    [True, False, True],
    # disc_iters=3.
    [True, False, False],
]


class ModularGanTest(parameterized.TestCase, test_utils.CompareGanTestCase):

  def setUp(self):
    super(ModularGanTest, self).setUp()
    self.model_dir = self._get_empty_model_dir()
    self.run_config = tf.contrib.tpu.RunConfig(
        model_dir=self.model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))

  def _runSingleTrainingStep(self, architecture, loss_fn, penalty_fn):
    parameters = {
        "architecture": architecture,
        "lambda": 1,
        "z_dim": 128,
    }
    with gin.unlock_config():
      gin.bind_parameter("penalty.fn", penalty_fn)
      gin.bind_parameter("loss.fn", loss_fn)
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=self.model_dir,
        conditional="biggan" in architecture)
    estimator = gan.as_estimator(self.run_config, batch_size=2, use_tpu=False)
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

  def testSingleTrainingStepWithJointGenForDisc(self):
    parameters = {
        "architecture": c.DUMMY_ARCH,
        "lambda": 1,
        "z_dim": 120,
        "disc_iters": 2,
    }
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=self.model_dir,
        experimental_joint_gen_for_disc=True,
        experimental_force_graph_unroll=True,
        conditional=True)
    estimator = gan.as_estimator(self.run_config, batch_size=2, use_tpu=False)
    estimator.train(gan.input_fn, steps=1)

  @parameterized.parameters([1, 2, 3])
  def testSingleTrainingStepDiscItersWithEma(self, disc_iters):
    parameters = {
        "architecture": c.DUMMY_ARCH,
        "lambda": 1,
        "z_dim": 128,
        "dics_iters": disc_iters,
    }
    gin.bind_parameter("ModularGAN.g_use_ema", True)
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=self.model_dir)
    estimator = gan.as_estimator(self.run_config, batch_size=2, use_tpu=False)
    estimator.train(gan.input_fn, steps=1)
    # Check for moving average variables in checkpoint.
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
    ema_vars = sorted([v[0] for v in tf.train.list_variables(checkpoint_path)
                       if v[0].endswith("ExponentialMovingAverage")])
    tf.logging.info("ema_vars=%s", ema_vars)
    expected_ema_vars = sorted([
        "generator/fc_noise/kernel/ExponentialMovingAverage",
        "generator/fc_noise/bias/ExponentialMovingAverage",
    ])
    self.assertAllEqual(ema_vars, expected_ema_vars)

  @parameterized.parameters(
      itertools.product([1, 2, 3], [False, True])
  )
  def testDiscItersIsUsedCorrectly(self, disc_iters, use_tpu):
    parameters = {
        "architecture": c.DUMMY_ARCH,
        "disc_iters": disc_iters,
        "lambda": 1,
        "z_dim": 128,
    }
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=self.model_dir,
        save_checkpoints_steps=1,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=self.model_dir)
    estimator = gan.as_estimator(run_config, batch_size=2, use_tpu=use_tpu)
    estimator.train(gan.input_fn, steps=3)

    disc_step_values = []
    gen_step_values = []

    for step in range(4):
      basename = os.path.join(self.model_dir, "model.ckpt-{}".format(step))
      self.assertTrue(tf.gfile.Exists(basename + ".index"))
      ckpt = tf.train.load_checkpoint(basename)

      disc_step_values.append(ckpt.get_tensor("global_step_disc"))
      gen_step_values.append(ckpt.get_tensor("global_step"))

    expected_disc_steps = np.arange(4) * disc_iters
    self.assertAllEqual(disc_step_values, expected_disc_steps)
    self.assertAllEqual(gen_step_values, [0, 1, 2, 3])


if __name__ == "__main__":
  tf.test.main()
