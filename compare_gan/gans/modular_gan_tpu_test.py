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

"""Tests TPU specfic parts of ModularGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
from compare_gan import datasets
from compare_gan import test_utils
from compare_gan.gans import consts as c
from compare_gan.gans.modular_gan import ModularGAN
import tensorflow as tf

FLAGS = flags.FLAGS


class ModularGanTpuTest(parameterized.TestCase, test_utils.CompareGanTestCase):

  def setUp(self):
    super(ModularGanTpuTest, self).setUp()
    self.model_dir = self._get_empty_model_dir()
    self.run_config = tf.contrib.tpu.RunConfig(
        model_dir=self.model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))

  @parameterized.parameters([1, 2, 5])
  def testBatchSize(self, disc_iters, use_tpu=True):
    parameters = {
        "architecture": c.DUMMY_ARCH,
        "lambda": 1,
        "z_dim": 128,
        "disc_iters": disc_iters,
    }
    batch_size = 16
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=self.model_dir)
    estimator = gan.as_estimator(self.run_config, batch_size=batch_size,
                                 use_tpu=True)
    estimator.train(gan.input_fn, steps=1)

    gen_args = gan.generator.call_arg_list
    disc_args = gan.discriminator.call_arg_list
    self.assertLen(gen_args, disc_iters + 1)  # D steps, G step.
    self.assertLen(disc_args, disc_iters + 1)  # D steps, G step.

    for args in gen_args:
      self.assertAllEqual(args["z"].shape.as_list(), [8, 128])
    for args in disc_args:
      self.assertAllEqual(args["x"].shape.as_list(), [16, 32, 32, 3])

  @parameterized.parameters([1, 2, 5])
  def testBatchSizeSplitDiscCalls(self, disc_iters):
    parameters = {
        "architecture": c.DUMMY_ARCH,
        "lambda": 1,
        "z_dim": 128,
        "disc_iters": disc_iters,
    }
    batch_size = 16
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        deprecated_split_disc_calls=True,
        model_dir=self.model_dir)
    estimator = gan.as_estimator(self.run_config, batch_size=batch_size,
                                 use_tpu=True)
    estimator.train(gan.input_fn, steps=1)

    gen_args = gan.generator.call_arg_list
    disc_args = gan.discriminator.call_arg_list
    self.assertLen(gen_args, disc_iters + 1)  # D steps, G step.
    # Each D and G step calls discriminator twice: for real and fake images.
    self.assertLen(disc_args, 2 * (disc_iters + 1))

    for args in gen_args:
      self.assertAllEqual(args["z"].shape.as_list(), [8, 128])
    for args in disc_args:
      self.assertAllEqual(args["x"].shape.as_list(), [8, 32, 32, 3])

  @parameterized.parameters([1, 2, 5])
  def testBatchSizeExperimentalJointGenForDisc(self, disc_iters):
    parameters = {
        "architecture": c.DUMMY_ARCH,
        "lambda": 1,
        "z_dim": 128,
        "disc_iters": disc_iters,
    }
    batch_size = 16
    dataset = datasets.get_dataset("cifar10")
    gan = ModularGAN(
        dataset=dataset,
        parameters=parameters,
        experimental_joint_gen_for_disc=True,
        model_dir=self.model_dir)
    estimator = gan.as_estimator(self.run_config, batch_size=batch_size,
                                 use_tpu=True)
    estimator.train(gan.input_fn, steps=1)

    gen_args = gan.generator.call_arg_list
    disc_args = gan.discriminator.call_arg_list
    self.assertLen(gen_args, 2)
    self.assertLen(disc_args, disc_iters + 1)

    self.assertAllEqual(gen_args[0]["z"].shape.as_list(), [8 * disc_iters, 128])
    self.assertAllEqual(gen_args[1]["z"].shape.as_list(), [8, 128])
    for args in disc_args:
      self.assertAllEqual(args["x"].shape.as_list(), [16, 32, 32, 3])


if __name__ == "__main__":
  tf.test.main()
