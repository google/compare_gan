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

"""Tests for eval_gan_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized

from compare_gan import datasets
from compare_gan import eval_gan_lib
from compare_gan import eval_utils
from compare_gan.gans import consts as c
from compare_gan.gans.modular_gan import ModularGAN
from compare_gan.metrics import fid_score
from compare_gan.metrics import fractal_dimension
from compare_gan.metrics import inception_score
from compare_gan.metrics import ms_ssim_score

import gin
import mock
import tensorflow as tf

FLAGS = flags.FLAGS


def create_fake_inception_graph():
  """Creates a `GraphDef` with that mocks the Inception V1 graph.

  It takes the input, multiplies it through a matrix full of 0.00001 values,
  and provides the results in the endpoints 'pool_3' and 'logits'. This
  matches the tensor names in the real Inception V1 model.
  the real inception model.

  Returns:
    `tf.GraphDef` for the mocked Inception V1 graph.
  """
  fake_inception = tf.Graph()
  with fake_inception.as_default():
    inputs = tf.placeholder(
        tf.float32, shape=[None, 299, 299, 3], name="Mul")
    w = tf.ones(shape=[299 * 299 * 3, 10]) * 0.00001
    outputs = tf.matmul(tf.layers.flatten(inputs), w)
    tf.identity(outputs, name="pool_3")
    tf.identity(outputs, name="logits")
  return fake_inception.as_graph_def()


class EvalGanLibTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(EvalGanLibTest, self).setUp()
    gin.clear_config()
    FLAGS.data_fake_dataset = True
    self.mock_get_graph = mock.patch.object(
        eval_utils, "get_inception_graph_def").start()
    self.mock_get_graph.return_value = create_fake_inception_graph()

  @parameterized.parameters(c.ARCHITECTURES)
  @flagsaver.flagsaver
  def test_end2end_checkpoint(self, architecture):
    """Takes real GAN (trained for 1 step) and evaluate it."""
    if architecture in {c.RESNET_STL_ARCH, c.RESNET30_ARCH}:
      # RESNET_STL_ARCH and RESNET107_ARCH do not support CIFAR image shape.
      return
    gin.bind_parameter("dataset.name", "cifar10")
    dataset = datasets.get_dataset("cifar10")
    options = {
        "architecture": architecture,
        "z_dim": 120,
        "disc_iters": 1,
        "lambda": 1,
    }
    model_dir = os.path.join(tf.test.get_temp_dir(), self.id())
    tf.logging.info("model_dir: %s" % model_dir)
    run_config = tf.contrib.tpu.RunConfig(model_dir=model_dir)
    gan = ModularGAN(dataset=dataset,
                     parameters=options,
                     conditional="biggan" in architecture,
                     model_dir=model_dir)
    estimator = gan.as_estimator(run_config, batch_size=2, use_tpu=False)
    estimator.train(input_fn=gan.input_fn, steps=1)
    export_path = os.path.join(model_dir, "tfhub")
    checkpoint_path = os.path.join(model_dir, "model.ckpt-1")
    module_spec = gan.as_module_spec()
    module_spec.export(export_path, checkpoint_path=checkpoint_path)

    eval_tasks = [
        fid_score.FIDScoreTask(),
        fractal_dimension.FractalDimensionTask(),
        inception_score.InceptionScoreTask(),
        ms_ssim_score.MultiscaleSSIMTask()
    ]
    result_dict = eval_gan_lib.evaluate_tfhub_module(
        export_path, eval_tasks, use_tpu=False, num_averaging_runs=1)
    tf.logging.info("result_dict: %s", result_dict)
    for score in ["fid_score", "fractal_dimension", "inception_score",
                  "ms_ssim"]:
      for stats in ["mean", "std", "list"]:
        required_key = "%s_%s" % (score, stats)
        self.assertIn(required_key, result_dict, "Missing: %s." % required_key)


if __name__ == "__main__":
  tf.test.main()
