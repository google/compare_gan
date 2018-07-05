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

"""Tests for GAN library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from compare_gan.src.gans import consts
from compare_gan.src.gans import GAN, WGAN, WGAN_GP, DRAGAN, LSGAN, BEGAN, VAE

import numpy as np
import tensorflow as tf

MODELS = {
    "GAN": GAN.GAN,
    "WGAN": WGAN.WGAN,
    "WGAN_GP": WGAN_GP.WGAN_GP,
    "DRAGAN": DRAGAN.DRAGAN,
    "LSGAN": LSGAN.LSGAN,
    "BEGAN": BEGAN.BEGAN,
    "VAE": VAE.VAE
}


class GANTest(tf.test.TestCase):

  def testGANBuildsAndImageShapeIsOk(self):
    features = np.random.randn(100, 2)
    labels = np.zeros(100)
    dataset_content = tf.data.Dataset.from_tensor_slices((features, labels))
    params = {
        # dataset params
        "input_height": 28,
        "input_width": 28,
        "output_height": 28,
        "output_width": 28,
        "c_dim": 1,
        "dataset_name": "mnist_fake",
        # training params
        "learning_rate": 0.0002,
        "beta1": 0.5,
        "z_dim": 62,
        "batch_size": 32,
        "training_steps": 200,
        "disc_iters": 5,
        "gamma": 0.1,
        "lambda": 0.1,
        "y_dim": 10,
        "pt_loss_weight": 1.0,
        "len_discrete_code": 10,
        "len_continuous_code": 2,
        "SUPERVISED": True,
        "discriminator_normalization": consts.NO_NORMALIZATION,
        "weight_clipping": -1.0,
        "save_checkpoint_steps": 5000
    }

    config = tf.ConfigProto(allow_soft_placement=True)
    for gan_type in MODELS:
      tf.reset_default_graph()

      runtime_info = collections.namedtuple(
          'RuntimeInfo', ['checkpoint_dir', 'result_dir', 'log_dir'])

      kwargs = dict(
          runtime_info=runtime_info,
          dataset_content=dataset_content,
          parameters=params)
      gan = MODELS[gan_type](**kwargs)
      gan.build_model()
      self.assertEqual(gan.fake_images.get_shape(), [32, 28, 28, 1])


if __name__ == "__main__":
  tf.test.main()
