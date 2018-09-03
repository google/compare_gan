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

"""Test cases to verify estimator training and matches our training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
from compare_gan.src import gan_lib

import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


class AbstractGANEstimatorTest(parameterized.TestCase, tf.test.TestCase):
  options = {
      'dataset': 'cifar10',
      'architecture': 'resnet_cifar_arch',
      'batch_size': 4,
      'discriminator_normalization': 'none',
      'gan_type': 'GAN_PENALTY',
      'optimizer': 'sgd',
      'penalty_type': 'no_penalty',
      'save_checkpoint_steps': 1,
      'training_steps': 5,
      'disc_iters': 3,
      'learning_rate': 0.000199999994948,
      'tf_seed': 42,
      'z_dim': 128,
      'lambda': 1,
      'beta2': 0.999000012875,
      'beta1': 0.5,
  }

  def load_checkpoint(self, checkpoint_dir, step):
    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt-' + str(step))
    if not tf.gfile.Exists(checkpoint_path + '.index'):
      checkpoint_path = os.path.join(checkpoint_dir,
                                     'GAN_PENALTY.model-' + str(step))
    return tf.train.load_checkpoint(checkpoint_path)

  # The test case trains a model with the non-estimator training loop and the
  # estimator training loop for a few steps and checks that the learned weights
  # match.
  @parameterized.named_parameters(
      [('DiscIters{}Unroll{}TPU{}'.format(*p), p[0], p[1], p[2]) for p in
       itertools.product([1, 3], [False, True], [False, True])]
  )
  @flagsaver.flagsaver
  def testEstimatorEqualsSession(self, disc_iters, unroll_disc_iters, use_tpu):
    if use_tpu and not unroll_disc_iters:
      # This is currently not supported. See b/111760885.
      return

    FLAGS.master = None
    FLAGS.iterations_per_loop = 1

    options = self.options.copy()
    options['disc_iters'] = disc_iters

    workdir = os.path.join(FLAGS.test_tmpdir, 'disc_iters{}/unroll{}'.format(
        disc_iters, unroll_disc_iters))

    options['use_estimator'] = False
    gan_lib.run_with_options(options, os.path.join(workdir, 'no_estimator'))

    # We use the first checkpoint of the run without Estimator to warm start.
    # Otherwise both models start with different values as the tf_seed is fixed
    # but the graphs are still different.
    warm_start = tf.estimator.WarmStartSettings(
        os.path.join(workdir, 'no_estimator/checkpoint/GAN_PENALTY.model-0'))
    options.update({
        'use_estimator': True,
        'unroll_disc_iters': unroll_disc_iters,
        'use_tpu': use_tpu,
    })
    gan_lib.run_with_options(options, os.path.join(workdir, 'estimator'),
                             warm_start_from=warm_start)

    for step in range(0, options['training_steps'], options['disc_iters']):
      tf.logging.info('Comparing checkpoints for step %d', step)
      estimator_ckpt = self.load_checkpoint(
          os.path.join(workdir, 'estimator/checkpoint'), step)
      baseline_ckpt = self.load_checkpoint(
          os.path.join(workdir, 'no_estimator/checkpoint'), step)

      not_equal = 0
      for name in baseline_ckpt.get_variable_to_shape_map():
        assert estimator_ckpt.has_tensor(name), name
        if name.endswith('moving_mean') or name.endswith('moving_variance'):
          # Ignore batch norm values.
          continue
        if 'discriminator' in name and name.endswith('biases'):

          continue
        t1 = estimator_ckpt.get_tensor(name)
        t2 = baseline_ckpt.get_tensor(name)
        if name == 'global_step':
          self.assertEqual(t1, step)
          self.assertEqual(t2, step)
          continue
        if not np.allclose(t1, t2, atol=1e-7):
          not_equal += 1
          diff = np.abs(t1 - t2)
          idx = np.argmax(diff)
          tf.logging.info(
              '%s is not close at step %d: maximum difference: %s (%s -> %s)',
              name, step, diff.max(), t1.flatten()[idx], t2.flatten()[idx])

      self.assertEqual(not_equal, 0)


if __name__ == '__main__':
  tf.test.main()
