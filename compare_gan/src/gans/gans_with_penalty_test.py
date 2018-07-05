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
from compare_gan.src.gans.gans_with_penalty import GAN_PENALTY, WGAN_PENALTY, LSGAN_PENALTY

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
    10 * [True],
    # disc_iters=2.
    5 * [True, False],
    # disc_iters=3.
    (4 * [True, False, False])[:10],
    # disc_iters=4.
    (3 * [True, False, False, False])[:10],
    # disc_iters=5.
    (3 * [True, False, False, False, False])[:10],
]


class FakeRuntimeInfo(object):

  def __init__(self, task_workdir=None):
    self.checkpoint_dir = task_workdir
    self.result_dir = task_workdir
    self.log_dir = task_workdir


class GANSWithPenaltyTest(parameterized.TestCase, tf.test.TestCase):
  parameters = {
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "z_dim": 100,
      "batch_size": 64,
      "training_steps": 10,
      "disc_iters": 1,
      "gamma": 0.1,
      "lambda": 0.1,
      "y_dim": 10,
      "pt_loss_weight": 1.0,
      "len_discrete_code": 10,
      "len_continuous_code": 2,
      "save_checkpoint_steps": 5000,
      "SUPERVISED": True,
      "weight_clipping": -1.0,
  }

  def isValidModel(self, model, normalization, architecture, penalty_type):
    if architecture == consts.SNDCGAN_ARCH:
      return normalization in [consts.SPECTRAL_NORM, consts.NO_NORMALIZATION]
    if architecture == consts.DCGAN_ARCH:
      return normalization in [consts.NO_NORMALIZATION, consts.SPECTRAL_NORM,
                               consts.BATCH_NORM]
    return True

  @parameterized.named_parameters(
      [(str(i), p[0], p[1], p[2], p[3]) for i, p in enumerate(
          itertools.product(TEST_MODELS, TEST_NORMALIZERS, TEST_ARCHITECTURES,
                            TEST_PENALTIES))])
  def testGANBuildsAndImageShapeIsOk(self, model, normalization, architecture,
                                     penalty_type):
    if not self.isValidModel(model, normalization, architecture, penalty_type):
      return
    parameters = self.parameters.copy()
    parameters.update(params.GetDatasetParameters("celeba"))
    parameters.update({
        "architecture": architecture,
        "penalty_type": penalty_type,
        "discriminator_normalization": normalization,
    })
    dataset_content = test_utils.load_fake_dataset(parameters).repeat()

    config = tf.ConfigProto(allow_soft_placement=True)
    tf.reset_default_graph()
    with tf.Session(config=config):
      kwargs = dict(
          runtime_info=FakeRuntimeInfo(),
          dataset_content=dataset_content,
          parameters=parameters)
      print("Testing loss %s, regularizer %s, with architecture %s." %
            (model, penalty_type, architecture))
      gan = model(**kwargs)
      gan.build_model()
      self.assertEqual(gan.fake_images.get_shape(), [64, 64, 64, 3])

  @parameterized.named_parameters([
      ("DiscIters" + str(i), i) for i in range(1, 6)
  ])
  def testDiscItersIsUsedCorrectly(self, disc_iters):
    parameters = self.parameters.copy()
    parameters.update(params.GetDatasetParameters("cifar10"))
    parameters.update({
        "batch_size": 2,
        "training_steps": 10,
        "save_checkpoint_steps": 1,
        "disc_iters": disc_iters,
        "architecture": consts.RESNET_CIFAR,
        "penalty_type": consts.NO_PENALTY,
        "discriminator_normalization": consts.NO_NORMALIZATION,
    })
    dataset_content = test_utils.load_fake_dataset(parameters).repeat()

    task_workdir = os.path.join(FLAGS.test_tmpdir, str(disc_iters))
    with tf.Graph().as_default(), tf.Session() as sess:
      gan = GAN_PENALTY(
          runtime_info=FakeRuntimeInfo(task_workdir),
          dataset_content=dataset_content,
          parameters=parameters)
      gan.build_model()
      gan.train(sess)

    # Read checkpoints for each training step. If the weight in the generator
    # changed we trained the generator during that step.
    previous_values = {}
    generator_trained = []
    for step in range(0, parameters["training_steps"] + 1):
      basename = os.path.join(task_workdir, "GAN_PENALTY.model-{}".format(step))
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
          if name.endswith('moving_mean') or name.endswith('moving_variance'):
            # Note: Even when we don't train the generator the batch norm
            # values still get updated.
            continue
          tf.logging.info("step %d: %s changed up to %f", step, name, diff)
          g_trained = True
      self.assertTrue(d_trained)  # Discriminator is trained every step.
      generator_trained.append(g_trained)

    self.assertEqual(generator_trained,
                     GENERATOR_TRAINED_IN_STEPS[disc_iters - 1])

  @parameterized.named_parameters([
      (str(i), p) for i, p in enumerate(
          TEST_ARCHITECTURES + [consts.RESNET_CIFAR])
  ])
  def testL2Regularization(self, architecture):
    parameters = self.parameters.copy()
    parameters.update(params.GetDatasetParameters("celeba"))
    parameters.update({
        "architecture": architecture,
        "penalty_type": consts.L2_PENALTY,
        "discriminator_normalization": consts.NO_NORMALIZATION,
    })
    dataset_content = test_utils.load_fake_dataset(parameters).repeat()

    config = tf.ConfigProto(allow_soft_placement=True)
    tf.reset_default_graph()
    with tf.Session(config=config):
      kwargs = dict(
          runtime_info=FakeRuntimeInfo(),
          dataset_content=dataset_content,
          parameters=parameters)
      gan = GAN_PENALTY(**kwargs)
      gan.build_model()


if __name__ == "__main__":
  tf.test.main()
