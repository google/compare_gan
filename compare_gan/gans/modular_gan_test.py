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

import datetime
import itertools
import os

from absl import flags
from absl.testing import parameterized
from compare_gan import datasets
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans import penalty_lib
from compare_gan.gans.modular_gan import ModularGAN
import gin
import numpy as np
from six.moves import range
import tensorflow as tf


FLAGS = flags.FLAGS
TEST_ARCHITECTURES = [c.INFOGAN_ARCH, c.DCGAN_ARCH, c.RESNET_CIFAR,
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


class ModularGANTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ModularGANTest, self).setUp()
    FLAGS.data_fake_dataset = True
    gin.clear_config()
    unused_sub_dir = str(datetime.datetime.now().microsecond)
    self.model_dir = os.path.join(FLAGS.test_tmpdir, unused_sub_dir)
    assert not tf.gfile.Exists(self.model_dir)
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
    self._runSingleTrainingStep(c.RESNET_CIFAR, loss_fn, penalty_lib.no_penalty)

  @parameterized.parameters(TEST_PENALTIES)
  def testSingleTrainingStepPenalties(self, penalty_fn):
    self._runSingleTrainingStep(c.RESNET_CIFAR, loss_lib.hinge, penalty_fn)

  def testSingleTrainingStepWithJointGenForDisc(self):
    parameters = {
        "architecture": c.RESNET5_BIGGAN_ARCH,
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
        conditional=True)
    estimator = gan.as_estimator(self.run_config, batch_size=2, use_tpu=False)
    estimator.train(gan.input_fn, steps=1)

  @parameterized.parameters([1, 2, 3])
  def testSingleTrainingStepDiscItersWithEma(self, disc_iters):
    parameters = {
        "architecture": c.RESNET_CIFAR,
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
        "generator/B1/up_conv_shortcut/kernel/ExponentialMovingAverage",
        "generator/B1/up_conv_shortcut/bias/ExponentialMovingAverage",
        "generator/B1/up_conv1/kernel/ExponentialMovingAverage",
        "generator/B1/up_conv1/bias/ExponentialMovingAverage",
        "generator/B1/same_conv2/kernel/ExponentialMovingAverage",
        "generator/B1/same_conv2/bias/ExponentialMovingAverage",
        "generator/B2/up_conv_shortcut/kernel/ExponentialMovingAverage",
        "generator/B2/up_conv_shortcut/bias/ExponentialMovingAverage",
        "generator/B2/up_conv1/kernel/ExponentialMovingAverage",
        "generator/B2/up_conv1/bias/ExponentialMovingAverage",
        "generator/B2/same_conv2/kernel/ExponentialMovingAverage",
        "generator/B2/same_conv2/bias/ExponentialMovingAverage",
        "generator/B3/up_conv_shortcut/kernel/ExponentialMovingAverage",
        "generator/B3/up_conv_shortcut/bias/ExponentialMovingAverage",
        "generator/B3/up_conv1/kernel/ExponentialMovingAverage",
        "generator/B3/up_conv1/bias/ExponentialMovingAverage",
        "generator/B3/same_conv2/kernel/ExponentialMovingAverage",
        "generator/B3/same_conv2/bias/ExponentialMovingAverage",
        "generator/final_conv/kernel/ExponentialMovingAverage",
        "generator/final_conv/bias/ExponentialMovingAverage",
    ])
    self.assertAllEqual(ema_vars, expected_ema_vars)

  @parameterized.parameters(
      itertools.product([1, 2, 3], [False, True])
  )
  def testDiscItersIsUsedCorrectly(self, disc_iters, use_tpu):

    return
    if disc_iters > 1 and use_tpu:

      return
    parameters = {
        "architecture": c.RESNET_CIFAR,
        "disc_iters": disc_iters,
        "lambda": 1,
        "z_dim": 128,
    }
    if not use_tpu:
      parameters["unroll_disc_iters"] = False
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

    # Read checkpoints for each training step. If the weight in the generator
    # changed we trained the generator during that step.
    previous_values = {}
    generator_trained = []
    for step in range(0, 4):
      basename = os.path.join(self.model_dir, "model.ckpt-{}".format(step))
      self.assertTrue(tf.gfile.Exists(basename + ".index"))
      ckpt = tf.train.load_checkpoint(basename)

      if step == 0:
        for name in ckpt.get_variable_to_shape_map():
          previous_values[name] = ckpt.get_tensor(name)
        continue

      d_trained = False
      g_trained = False
      for name in ckpt.get_variable_to_shape_map():
        t = ckpt.get_tensor(name)
        diff = np.abs(previous_values[name] - t).max()
        previous_values[name] = t
        if "discriminator" in name and diff > 1e-10:
          d_trained = True
        elif "generator" in name and diff > 1e-10:
          if name.endswith("moving_mean") or name.endswith("moving_variance"):
            # Note: Even when we don't train the generator the batch norm
            # values still get updated.
            continue
          tf.logging.info("step %d: %s changed up to %f", step, name, diff)
          g_trained = True
      self.assertTrue(d_trained)  # Discriminator is trained every step.
      generator_trained.append(g_trained)

    self.assertEqual(generator_trained,
                     GENERATOR_TRAINED_IN_STEPS[disc_iters - 1])


if __name__ == "__main__":
  tf.test.main()
