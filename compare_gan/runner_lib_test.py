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

"""Tests high level behavior of the runner_lib.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized

from compare_gan import eval_gan_lib
from compare_gan import runner_lib
from compare_gan import test_utils
from compare_gan.architectures import arch_ops
from compare_gan.gans.modular_gan import ModularGAN

import gin
import numpy as np
from six.moves import range
import tensorflow as tf


FLAGS = flags.FLAGS


class RunnerLibTest(parameterized.TestCase, test_utils.CompareGanTestCase):

  @parameterized.named_parameters([
      ("SameSeeds", 42, 42),
      ("DifferentSeeds", 1, 42),
      ("NoSeeds", None, None),
  ])
  def testWeightInitialization(self, seed1, seed2):
    gin.bind_parameter("dataset.name", "cifar10")
    gin.bind_parameter("ModularGAN.g_optimizer_fn",
                       tf.train.GradientDescentOptimizer)
    options = {
        "architecture": "resnet_cifar_arch",
        "batch_size": 2,
        "disc_iters": 1,
        "gan_class": ModularGAN,
        "lambda": 1,
        "training_steps": 1,
        "z_dim": 128,
    }
    work_dir = self._get_empty_model_dir()
    seeds = [seed1, seed2]
    for i in range(2):
      model_dir = os.path.join(work_dir, str(i))
      seed = seeds[i]
      run_config = tf.contrib.tpu.RunConfig(
          model_dir=model_dir, tf_random_seed=seed)
      task_manager = runner_lib.TaskManager(model_dir)
      runner_lib.run_with_schedule(
          "train",
          run_config=run_config,
          task_manager=task_manager,
          options=options,
          use_tpu=False)

    checkpoint_path_0 = os.path.join(work_dir, "0/model.ckpt-0")
    checkpoint_path_1 = os.path.join(work_dir, "1/model.ckpt-0")
    checkpoint_reader_0 = tf.train.load_checkpoint(checkpoint_path_0)
    checkpoint_reader_1 = tf.train.load_checkpoint(checkpoint_path_1)
    for name, _ in tf.train.list_variables(checkpoint_path_0):
      tf.logging.info(name)
      t0 = checkpoint_reader_0.get_tensor(name)
      t1 = checkpoint_reader_1.get_tensor(name)
      zero_initialized_vars = [
          "bias", "biases", "beta", "moving_mean", "global_step",
          "global_step_disc"
      ]
      one_initialized_vars = ["gamma", "moving_variance"]
      if any(name.endswith(e) for e in zero_initialized_vars):
        # Variables that are always initialized to 0.
        self.assertAllClose(t0, np.zeros_like(t0))
        self.assertAllClose(t1, np.zeros_like(t1))
      elif any(name.endswith(e) for e in one_initialized_vars):
        # Variables that are always initialized to 1.
        self.assertAllClose(t0, np.ones_like(t0))
        self.assertAllClose(t1, np.ones_like(t1))
      elif seed1 is not None and seed1 == seed2:
        # Same random seed.
        self.assertAllClose(t0, t1)
      else:
        # Different random seeds.
        logging.info("name=%s, t0=%s, t1=%s", name, t0, t1)
        self.assertNotAllClose(t0, t1)

  @parameterized.named_parameters([
      ("WithRealData", False),
      ("WithFakeData", True),
  ])
  @flagsaver.flagsaver
  def testTrainingIsDeterministic(self, fake_dataset):
    FLAGS.data_fake_dataset = fake_dataset
    gin.bind_parameter("dataset.name", "cifar10")
    options = {
        "architecture": "resnet_cifar_arch",
        "batch_size": 2,
        "disc_iters": 1,
        "gan_class": ModularGAN,
        "lambda": 1,
        "training_steps": 3,
        "z_dim": 128,
    }
    work_dir = self._get_empty_model_dir()
    for i in range(2):
      model_dir = os.path.join(work_dir, str(i))
      run_config = tf.contrib.tpu.RunConfig(
          model_dir=model_dir, tf_random_seed=3)
      task_manager = runner_lib.TaskManager(model_dir)
      runner_lib.run_with_schedule(
          "train",
          run_config=run_config,
          task_manager=task_manager,
          options=options,
          use_tpu=False,
          num_eval_averaging_runs=1)

    checkpoint_path_0 = os.path.join(work_dir, "0/model.ckpt-3")
    checkpoint_path_1 = os.path.join(work_dir, "1/model.ckpt-3")
    checkpoint_reader_0 = tf.train.load_checkpoint(checkpoint_path_0)
    checkpoint_reader_1 = tf.train.load_checkpoint(checkpoint_path_1)
    for name, _ in tf.train.list_variables(checkpoint_path_0):
      tf.logging.info(name)
      t0 = checkpoint_reader_0.get_tensor(name)
      t1 = checkpoint_reader_1.get_tensor(name)
      self.assertAllClose(t0, t1, msg=name)

  @parameterized.parameters([
      {"use_tpu": False},
      # {"use_tpu": True},
  ])
  def testTrainAndEval(self, use_tpu):
    gin.bind_parameter("dataset.name", "cifar10")
    options = {
        "architecture": "resnet_cifar_arch",
        "batch_size": 2,
        "disc_iters": 1,
        "gan_class": ModularGAN,
        "lambda": 1,
        "training_steps": 1,
        "z_dim": 128,
    }
    model_dir = self._get_empty_model_dir()
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    task_manager = runner_lib.TaskManager(model_dir)
    runner_lib.run_with_schedule(
        "eval_after_train",
        run_config=run_config,
        task_manager=task_manager,
        options=options,
        use_tpu=use_tpu,
        num_eval_averaging_runs=1,
        eval_every_steps=None)
    expected_files = [
        "TRAIN_DONE", "checkpoint", "model.ckpt-0.data-00000-of-00001",
        "model.ckpt-0.index", "model.ckpt-0.meta",
        "model.ckpt-1.data-00000-of-00001", "model.ckpt-1.index",
        "model.ckpt-1.meta", "operative_config-0.gin", "tfhub"]
    self.assertAllInSet(expected_files, tf.gfile.ListDirectory(model_dir))

  def testTrainAndEvalWithSpectralNormAndEma(self):
    gin.bind_parameter("dataset.name", "cifar10")
    gin.bind_parameter("ModularGAN.g_use_ema", True)
    gin.bind_parameter("G.spectral_norm", True)
    options = {
        "architecture": "resnet_cifar_arch",
        "batch_size": 2,
        "disc_iters": 1,
        "gan_class": ModularGAN,
        "lambda": 1,
        "training_steps": 1,
        "z_dim": 128,
    }
    model_dir = self._get_empty_model_dir()
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    task_manager = runner_lib.TaskManager(model_dir)
    runner_lib.run_with_schedule(
        "eval_after_train",
        run_config=run_config,
        task_manager=task_manager,
        options=options,
        use_tpu=False,
        num_eval_averaging_runs=1,
        eval_every_steps=None)
    expected_files = [
        "TRAIN_DONE", "checkpoint", "model.ckpt-0.data-00000-of-00001",
        "model.ckpt-0.index", "model.ckpt-0.meta",
        "model.ckpt-1.data-00000-of-00001", "model.ckpt-1.index",
        "model.ckpt-1.meta", "operative_config-0.gin", "tfhub"]
    self.assertAllInSet(expected_files, tf.gfile.ListDirectory(model_dir))

  def testTrainAndEvalWithBatchNormAccu(self):
    gin.bind_parameter("dataset.name", "cifar10")
    gin.bind_parameter("standardize_batch.use_moving_averages", False)
    gin.bind_parameter("G.batch_norm_fn", arch_ops.batch_norm)
    options = {
        "architecture": "resnet_cifar_arch",
        "batch_size": 2,
        "disc_iters": 1,
        "gan_class": ModularGAN,
        "lambda": 1,
        "training_steps": 1,
        "z_dim": 128,
    }
    model_dir = self._get_empty_model_dir()
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    task_manager = runner_lib.TaskManager(model_dir)
    # Wrap _UpdateBnAccumulators to only perform one accumulator update step.
    # Otherwise the test case would time out.
    orig_update_bn_accumulators = eval_gan_lib._update_bn_accumulators
    def mock_update_bn_accumulators(sess, generated, num_accu_examples):
      del num_accu_examples
      return orig_update_bn_accumulators(sess, generated, num_accu_examples=64)
    eval_gan_lib._update_bn_accumulators = mock_update_bn_accumulators
    runner_lib.run_with_schedule(
        "eval_after_train",
        run_config=run_config,
        task_manager=task_manager,
        options=options,
        use_tpu=False,
        num_eval_averaging_runs=1,
        eval_every_steps=None)
    expected_tfhub_files = [
        "checkpoint", "model-with-accu.ckpt.data-00000-of-00001",
        "model-with-accu.ckpt.index", "model-with-accu.ckpt.meta"]
    self.assertAllInSet(
        expected_tfhub_files,
        tf.gfile.ListDirectory(os.path.join(model_dir, "tfhub/0")))


if __name__ == "__main__":
  tf.test.main()
