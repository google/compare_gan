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

import os

from absl import flags
from absl import logging
from absl.testing import parameterized
from compare_gan import datasets
from compare_gan.gans import consts as c
from compare_gan.gans.modular_gan import ModularGAN
import gin
import tensorflow as tf


FLAGS = flags.FLAGS


class MockModularGAN(ModularGAN):

  def __init__(self, **kwargs):
    super(MockModularGAN, self).__init__(**kwargs)
    self.gen_args = []
    self.disc_args = []

  def generator(self, z, y, **kwargs):
    self.gen_args.append(dict(z=z, y=y, **kwargs))
    return super(MockModularGAN, self).generator(z=z, y=y, **kwargs)

  def discriminator(self, x, y, **kwargs):
    self.disc_args.append(dict(x=x, y=y, **kwargs))
    return super(MockModularGAN, self).discriminator(x=x, y=y, **kwargs)


class ModularGANTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ModularGANTest, self).setUp()
    FLAGS.data_fake_dataset = True
    gin.clear_config()
    self.model_dir = os.path.join(FLAGS.test_tmpdir, "model_dir")
    if tf.gfile.Exists(self.model_dir):
      tf.gfile.DeleteRecursively(self.model_dir)
    self.run_config = tf.contrib.tpu.RunConfig(
        model_dir=self.model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))

  @parameterized.parameters([1, 2, 5])
  def testBatchSize(self, disc_iters, use_tpu=True):
    parameters = {
        "architecture": c.RESNET5_ARCH,
        "lambda": 1,
        "z_dim": 128,
        "disc_iters": disc_iters,
    }
    batch_size = 16
    dataset = datasets.get_dataset("cifar10")
    gan = MockModularGAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=self.model_dir)
    estimator = gan.as_estimator(self.run_config, batch_size=batch_size,
                                 use_tpu=use_tpu)
    estimator.train(gan.input_fn, steps=1)
    logging.info("gen_args: %s", "\n".join(str(a) for a in gan.gen_args))
    logging.info("disc_args: %s", "\n".join(str(a) for a in gan.disc_args))
    num_shards = 2 if use_tpu else 1
    assert batch_size % num_shards == 0
    gen_bs = batch_size // num_shards
    disc_bs = gen_bs * 2  # merged discriminator calls.
    self.assertLen(gan.gen_args, disc_iters + 2)
    for args in gan.gen_args:
      self.assertAllEqual(args["z"].shape.as_list(), [gen_bs, 128])
    self.assertLen(gan.disc_args, disc_iters + 1)
    for args in gan.disc_args:
      self.assertAllEqual(args["x"].shape.as_list(), [disc_bs, 32, 32, 3])


if __name__ == "__main__":
  tf.test.main()
