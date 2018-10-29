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

import itertools
import os

from absl import flags
from absl.testing import parameterized

from compare_gan.src import params
from compare_gan.src import test_utils
from compare_gan.src.gans import consts
from compare_gan.src.gans.gans_with_penalty import GAN_PENALTY
from compare_gan.src.gans.gans_with_penalty import LSGAN_PENALTY
from compare_gan.src.gans.gans_with_penalty import WGAN_PENALTY

import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

TEST_MODELS = [GAN_PENALTY, WGAN_PENALTY, LSGAN_PENALTY]
TEST_NORMALIZERS = consts.NORMALIZERS
TEST_PENALTIES = consts.PENALTIES
TEST_ARCHITECTURES = [
    consts.INFOGAN_ARCH, consts.DCGAN_ARCH, consts.SNDCGAN_ARCH,
    consts.RESNET5_ARCH
]
GENERATOR_TRAINED_IN_STEPS = [
    # disc_iters=1.
    5 * [True],
    # disc_iters=2.
    (3 * [True, False])[:5],
    # disc_iters=3.
    (3 * [True, False, False])[:5],
]


class FakeRuntimeInfo(object):

  def __init__(self, model_dir=None):
    self.checkpoint_dir = model_dir
    self.result_dir = model_dir
    self.log_dir = model_dir


class GANSWithPenaltyEstimatorTest(parameterized.TestCase, tf.test.TestCase):
  parameters = {
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "z_dim": 64,
      "batch_size": 2,
      "gamma": 0.1,
      "lambda": 0.1,
      "y_dim": 10,
      "pt_loss_weight": 1.0,
      "len_discrete_code": 10,
      "len_continuous_code": 2,
      "save_checkpoint_steps": 1,
      "weight_clipping": -1.0,
      "tf_seed": 42,
      "use_estimator": True,
  }

  def isValidModel(self, model, normalization, architecture, penalty_type):
    if architecture == consts.SNDCGAN_ARCH:
      return normalization in [consts.SPECTRAL_NORM, consts.NO_NORMALIZATION]
    if architecture == consts.DCGAN_ARCH:
      return normalization in [
          consts.NO_NORMALIZATION, consts.SPECTRAL_NORM, consts.BATCH_NORM
      ]
    return True

  @parameterized.named_parameters(
      [["m{}norm{}arch{}penalty{}".format(*p)] + list(p)
       for p in itertools.product(TEST_MODELS, TEST_NORMALIZERS,
                                  TEST_ARCHITECTURES, TEST_PENALTIES)])
  def testSingleTrainingStepOnTPU(self, model, normalization, architecture,
                                  penalty_type):
    if not self.isValidModel(model, normalization, architecture, penalty_type):
      return
    parameters = self.parameters.copy()
    parameters.update(params.GetDatasetParameters("cifar10"))
    parameters.update({
        "use_tpu": True,
        "discriminator_normalization": normalization,
        "architecture": architecture,
        "penalty_type": penalty_type,
        "disc_iters": 1,
        "training_steps": 1,
    })
    dataset_content = test_utils.load_fake_dataset(parameters).repeat()

    model_dir = os.path.join(FLAGS.test_tmpdir, model.__name__, normalization,
                             architecture, penalty_type)

    config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    print("Testing loss %s, regularizer %s, with architecture %s." %
          (model, penalty_type, architecture))
    gan = model(
        runtime_info=FakeRuntimeInfo(model_dir),
        dataset_content=dataset_content,
        parameters=parameters)
    gan.train_with_estimator(config)

  @parameterized.named_parameters(
      [["disc{}tpu{}".format(*p)] + list(p)
       for p in itertools.product(range(1, 4), [False, True])])
  def testDiscItersIsUsedCorrectly(self, disc_iters, use_tpu):
    if disc_iters > 1 and use_tpu:

      return
    parameters = self.parameters.copy()
    parameters.update(params.GetDatasetParameters("cifar10"))
    parameters.update({
        "use_tpu": use_tpu,
        "discriminator_normalization": consts.NO_NORMALIZATION,

        "architecture": consts.RESNET_CIFAR,
        "penalty_type": consts.NO_PENALTY,
        "disc_iters": disc_iters,
        "training_steps": 5,
    })
    dataset_content = test_utils.load_fake_dataset(parameters).repeat()

    model_dir = os.path.join(FLAGS.test_tmpdir, str(disc_iters))

    config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=1,
        keep_checkpoint_max=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    gan = GAN_PENALTY(
        runtime_info=FakeRuntimeInfo(model_dir),
        dataset_content=dataset_content,
        parameters=parameters)
    gan.train_with_estimator(config)

    # Read checkpoints for each training step. If the weight in the generator
    # changed we trained the generator during that step.
    previous_values = {}
    generator_trained = []
    for step in range(0, parameters["training_steps"] + 1):
      basename = os.path.join(model_dir, "model.ckpt-{}".format(step))
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
