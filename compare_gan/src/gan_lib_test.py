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

"""Tests for compare_gan.gan_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import string

from absl import flags
from absl.testing import parameterized

from compare_gan.src import gan_lib
from compare_gan.src.gans import consts

import numpy as np
from six.moves import range
import tensorflow as tf

FLAGS = flags.FLAGS


class GanLibTest(parameterized.TestCase, tf.test.TestCase):

  def testLoadingTriangles(self):
    with tf.Graph().as_default():
      iterator = gan_lib.load_dataset("triangles").batch(
          32).make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (32, 28, 28, 1))
        self.assertEqual(label.shape, (32,))
        self.assertEqual(label[4], 3)
    with tf.Graph().as_default():
      iterator = gan_lib.load_dataset(
          "triangles", split_name="test").make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (28, 28, 1))
        self.assertEqual(label.shape, ())
    with tf.Graph().as_default():
      iterator = gan_lib.load_dataset(
          "triangles", split_name="val").make_one_shot_iterator(
              ).get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (28, 28, 1))
        self.assertEqual(label.shape, ())

  def testLoadingMnist(self):
    with tf.Graph().as_default():
      dataset = gan_lib.load_dataset("mnist")
      iterator = dataset.make_one_shot_iterator().get_next()
      with tf.Session() as sess:
        image, label = sess.run(iterator)
        self.assertEqual(image.shape, (28, 28, 1))
        self.assertEqual(label.shape, ())

  def trainSingleStep(self, tf_seed):
    """Train a GAN for a single training step and return the checkpoint."""
    parameters = {
        "tf_seed": tf_seed,
        "learning_rate": 0.0002,
        "z_dim": 64,
        "batch_size": 2,
        "training_steps": 1,
        "disc_iters": 1,
        "save_checkpoint_steps": 5000,
        "discriminator_normalization": consts.NO_NORMALIZATION,
        "dataset": "fake",
        "gan_type": "GAN",
        "penalty_type": consts.NO_PENALTY,
        "architecture": consts.RESNET_CIFAR,
        "lambda": 0.1,
    }
    random.seed(None)
    exp_name = ''.join(random.choice(string.ascii_uppercase) for _ in range(16))
    task_workdir = os.path.join(FLAGS.test_tmpdir, exp_name)
    gan_lib.run_with_options(parameters, task_workdir)
    ckpt_fn = os.path.join(task_workdir, "checkpoint/{}.model-0".format(
                           parameters["gan_type"]))
    tf.logging.info("ckpt_fn: %s", ckpt_fn)
    self.assertTrue(tf.gfile.Exists(ckpt_fn + ".index"))
    return tf.train.load_checkpoint(ckpt_fn)

  def testSameTFRandomSeed(self):
    # Setting the same tf_seed should give the same initial values at each run.
    # In practice training still converge due to non-deterministic behavior
    # in certain operations on the hardware level (e.g. cuDNN optimizations).
    ckpt_1 = self.trainSingleStep(tf_seed=42)
    ckpt_2 = self.trainSingleStep(tf_seed=42)

    for name in ckpt_1.get_variable_to_shape_map():
      self.assertTrue(ckpt_2.has_tensor(name))
      t1 = ckpt_1.get_tensor(name)
      t2 = ckpt_2.get_tensor(name)
      np.testing.assert_almost_equal(t1, t2)

  @parameterized.named_parameters([
      ('Given', 1, 2),
      ('OneNone', 1, None),
      ('BothNone', None, None),
  ])
  def testDifferentTFRandomSeed(self, seed_1, seed_2):
    ckpt_1 = self.trainSingleStep(tf_seed=seed_1)
    ckpt_2 = self.trainSingleStep(tf_seed=seed_2)

    diff_counter = 0
    for name in ckpt_1.get_variable_to_shape_map():
      self.assertTrue(ckpt_2.has_tensor(name))
      t1 = ckpt_1.get_tensor(name)
      t2 = ckpt_2.get_tensor(name)
      if np.abs(t1 - t2).sum() > 0:
         diff_counter += 1
    self.assertGreater(diff_counter, 0)


if __name__ == "__main__":
  tf.test.main()
