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

"""Run one task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from compare_gan.src import eval_gan_lib
from compare_gan.src import gan_lib
from compare_gan.src import params
from compare_gan.src import task_utils
from compare_gan.src.gans import consts
from compare_gan.src.multi_gan import dataset
from compare_gan.src.multi_gan import multi_gan
from compare_gan.src.multi_gan import multi_gan_background

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def GetGANGridSearch(dataset_name, training_steps, num_seeds):
  """Standard GAN grid search used in the paper."""

  config = {  # 48 x num_seeds workers
      "gan_type": [consts.GAN_WITH_PENALTY, consts.WGAN_WITH_PENALTY],
      "penalty_type": [consts.NO_PENALTY, consts.WGANGP_PENALTY],
      "discriminator_normalization": [
          consts.NO_NORMALIZATION, consts.SPECTRAL_NORM],
      "architecture": consts.DCGAN_ARCH,
      "dataset": dataset_name,
      "tf_seed": list(range(num_seeds)),
      "training_steps": training_steps,
      "save_checkpoint_steps": 20000,
      "batch_size": 64,
      "optimizer": "adam",
      "z_dim": 64,

      "__initial_trials": json.dumps(task_utils.CrossProduct({
          "learning_rate": 0.0001,
          "lambda": [1, 10],
          ("beta1", "beta2", "disc_iters"): [
              (0.5, 0.9, 5), (0.5, 0.999, 5), (0.9, 0.999, 5)],
      })),
  }

  return config


def GetMultiGANGridSearch(dataset_name, training_steps, num_seeds, aggregate):
  """Standard MultiGAN grid search used in the paper."""

  config = {  # 42 x num_seeds workers
      "gan_type": "MultiGAN",
      "penalty_type": consts.WGANGP_PENALTY,
      "discriminator_normalization": [
          consts.NO_NORMALIZATION, consts.SPECTRAL_NORM],
      "architecture": consts.DCGAN_ARCH,
      "dataset": dataset_name,
      "tf_seed": list(range(num_seeds)),
      "training_steps": training_steps,
      "save_checkpoint_steps": 20000,
      "batch_size": 64,
      "optimizer": "adam",

      # Model params.
      "aggregate": aggregate,
      "__initial_trials": json.dumps(task_utils.CrossProduct({
          ("n_blocks", "share_block_weights", "n_heads", "k"): [
              (0, False, 0, 3), (0, False, 0, 4),  # M-GAN [3, 4]
              (0, False, 0, 5),                    # M-GAN  5
              (1, False, 1, 3), (1, False, 2, 3),  # RM-GAN 3
              (2, False, 1, 3), (2, True, 1, 3),   # RM-GAN 3
              (2, False, 2, 3), (2, True, 2, 3),   # RM-GAN 3
              (1, False, 1, 4), (1, False, 2, 4),  # RM-GAN 4
              (2, False, 1, 4), (2, True, 1, 4),   # RM-GAN 4
              (2, False, 2, 4), (2, True, 2, 4),   # RM-GAN 4
              (1, False, 1, 5), (1, False, 2, 5),  # RM-GAN 5
              (2, False, 1, 5), (2, True, 1, 5),   # RM-GAN 5
              (2, False, 2, 5), (2, True, 2, 5),   # RM-GAN 5
          ],
          ("z_dim", "embedding_dim"): [(64, 32)],
          "learning_rate": 0.0001,
          "lambda": 1,
          ("beta1", "beta2", "disc_iters"): [(0.9, 0.999, 5)],
      }))
  }

  return config


def GetMultiGANBackgroundGridSearch(dataset_name, training_steps, num_seeds,
                                    aggregate):
  """Standard MultiGAN grid search used in the paper."""

  config = GetMultiGANGridSearch(  # 84 x num_seeds workers
      dataset_name, training_steps, num_seeds, aggregate)
  config["gan_type"] = "MultiGANBackground"
  config["background_interaction"] = [True, False]

  return config


def GetMultiGANGridSearchKPlusOne(dataset_name, training_steps, num_seeds,
                                  aggregate):
  """Standard MultiGAN grid search with k+1 used to compare against MBG."""

  config = {  # 42 x num_seeds workers
      "gan_type": "MultiGAN",
      "penalty_type": consts.WGANGP_PENALTY,
      "discriminator_normalization": [
          consts.NO_NORMALIZATION, consts.SPECTRAL_NORM],
      "architecture": consts.DCGAN_ARCH,
      "dataset": dataset_name,
      "tf_seed": list(range(num_seeds)),
      "training_steps": training_steps,
      "save_checkpoint_steps": 20000,
      "batch_size": 64,
      "optimizer": "adam",

      # Model params.
      "aggregate": aggregate,
      "__initial_trials": json.dumps(task_utils.CrossProduct({
          ("n_blocks", "share_block_weights", "n_heads", "k"): [
              (0, False, 0, 4), (0, False, 0, 5),  # M-GAN [4, 5]
              (0, False, 0, 6),                    # M-GAN  6
              (1, False, 1, 4), (1, False, 2, 4),  # RM-GAN 4
              (2, False, 1, 4), (2, True, 1, 4),   # RM-GAN 4
              (2, False, 2, 4), (2, True, 2, 4),   # RM-GAN 4
              (1, False, 1, 5), (1, False, 2, 5),  # RM-GAN 5
              (2, False, 1, 5), (2, True, 1, 5),   # RM-GAN 5
              (2, False, 2, 5), (2, True, 2, 5),   # RM-GAN 5
              (1, False, 1, 6), (1, False, 2, 6),  # RM-GAN 6
              (2, False, 1, 6), (2, True, 1, 6),   # RM-GAN 6
              (2, False, 2, 6), (2, True, 2, 6),   # RM-GAN 6
          ],
          ("z_dim", "embedding_dim"): [(64, 32)],
          "learning_rate": 0.0001,
          "lambda": 1,
          ("beta1", "beta2", "disc_iters"): [(0.9, 0.999, 5)],
      }))
  }

  return config


def GetMultiGANBackgroundRandomSearch(dataset_name, training_steps, aggregate,
                                      num_tasks):
  """MultiGANBackground random search used in the paper."""

  config = {
      "gan_type": "MultiGANBackground",
      "architecture": consts.DCGAN_ARCH,
      "dataset": dataset_name,
      "tf_seed": 1,
      "training_steps": training_steps,
      "save_checkpoint_steps": 20000,
      "batch_size": 64,
      "optimizer": "adam",

      # Model params.
      "aggregate": aggregate,
      "n_blocks": 2,
      "share_block_weights": False,
      "n_heads": 2,
      "k": 5,
      "z_dim": 64,
      "embedding_dim": 32,
      "background_interaction": False,
      "__num_tasks": num_tasks
  }

  return config


def GetMetaTasks(experiment_name):
  """Returns meta options to be used for study generation.

  Args:
    experiment_name: name of an experiment

  Raises:
    ValueError: When experiment is not found.
  """

  if experiment_name == "multi_gan-debug":
    meta_config = {
        "gan_type": ["MultiGAN",],
        "penalty_type": [consts.WGANGP_PENALTY],
        "discriminator_normalization": [consts.SPECTRAL_NORM],
        "architecture": consts.DCGAN_ARCH,
        "dataset": ["multi-mnist-3-uniform"],
        "sampler": ["rs",],
        "tf_seed": [0],
        "training_steps": [1000],
        "save_checkpoint_steps": [100],
        "batch_size": [64],
        "optimizer": ["adam"],
        "learning_rate": [0.0001],
        "lambda": [10],
        "beta1": [0.5],
        "beta2": [0.9],
        "disc_iters": [5],

        # Model params.
        "k": [3],
        "aggregate": ["sum_clip"],
        "n_heads": 4,
        "n_blocks": 2,
        "share_block_weights": True,
        "embedding_dim": 32,
    }

  #######################
  ## PAPER EXPERIMENTS ##
  #######################

  # MULTI-MNIST

  # 240
  elif experiment_name == "gan-base-experiment-paper":
    meta_config = GetGANGridSearch(
        dataset_name="multi-mnist-3-uniform",
        training_steps=1000000, num_seeds=5)

  # 210
  elif experiment_name == "multi_gan-base-experiment-paper":
    meta_config = GetMultiGANGridSearch(
        dataset_name="multi-mnist-3-uniform",
        training_steps=1000000, num_seeds=5, aggregate="sum_clip")

  # 480
  elif experiment_name == "gan-relational-experiment-paper":
    meta_config = GetGANGridSearch(
        dataset_name=[
            "multi-mnist-3-triplet", "multi-mnist-3-uniform-rgb-occluded"],
        training_steps=1000000, num_seeds=5)

  # 210
  elif experiment_name == "multi_gan-relational-experiment-triplet-paper":
    meta_config = GetMultiGANGridSearch(
        dataset_name="multi-mnist-3-triplet",
        training_steps=1000000, num_seeds=5, aggregate="sum_clip")

  # 210
  elif experiment_name == "multi_gan-relational-experiment-rgb-occluded-paper":
    meta_config = GetMultiGANGridSearch(
        dataset_name="multi-mnist-3-uniform-rgb-occluded",
        training_steps=1000000, num_seeds=5, aggregate="implicit_alpha")

  # CIFAR 10

  # 240
  elif experiment_name == "gan-background-experiment-paper":
    meta_config = GetGANGridSearch(
        dataset_name="multi-mnist-3-uniform-rgb-occluded-cifar10",
        training_steps=1000000, num_seeds=5)

  # 210
  elif experiment_name == "multi_gan-background-experiment-paper":
    meta_config = GetMultiGANGridSearchKPlusOne(
        dataset_name="multi-mnist-3-uniform-rgb-occluded-cifar10",
        training_steps=1000000, num_seeds=5, aggregate="alpha")

  # 420
  elif experiment_name == "multi_gan_bg-background-experiment-paper":
    meta_config = GetMultiGANBackgroundGridSearch(
        dataset_name="multi-mnist-3-uniform-rgb-occluded-cifar10",
        training_steps=1000000, num_seeds=5, aggregate="alpha")

  # CLEVR

  # 240
  elif experiment_name == "gan-clevr-experiment-paper":
    meta_config = GetGANGridSearch(
        dataset_name="clevr", training_steps=1000000, num_seeds=5)

  # 210
  elif experiment_name == "multi_gan-clevr-experiment-paper":
    meta_config = GetMultiGANGridSearchKPlusOne(
        dataset_name="clevr", training_steps=1000000, num_seeds=5,
        aggregate="alpha")

  # 420
  elif experiment_name == "multi_gan_bg-clevr-experiment-paper":
    meta_config = GetMultiGANBackgroundGridSearch(
        dataset_name="clevr", training_steps=1000000, num_seeds=5,
        aggregate="alpha")

  # 200
  elif experiment_name == "multi_gan_bg-clevr-rs200k-paper":
    meta_config = GetMultiGANBackgroundRandomSearch(
        dataset_name="clevr", training_steps=200000, aggregate="alpha",
        num_tasks=200)

  else:
    raise ValueError("Unknown study-based experiment %s." % experiment_name)

  options = task_utils.CrossProduct(meta_config)
  return options


def AddGansAndDatasets():
  """Injects MultiGAN models, parameters and datasets.

  This code injects the GAN model and its default parameters to the framework.
  Must be run just after the main.
  """
  gan_lib.MODELS.update({
      "MultiGAN": multi_gan.MultiGAN,
      "MultiGANBackground": multi_gan_background.MultiGANBackground
  })
  params.PARAMETERS.update({
      "MultiGAN": multi_gan.MultiGANHyperParams,
      "MultiGANBackground": multi_gan_background.MultiGANBackgroundHyperParams
  })
  eval_gan_lib.SUPPORTED_GANS.extend(["MultiGAN", "MultiGANBackground"])
  eval_gan_lib.DEFAULT_VALUES.update({
      "k": -1,
      "aggregate": "none",
      "embedding_dim": -1,
      "n_blocks": -1,
      "share_block_weights": False,
      "n_heads": -1,
      "background_interaction": False,
  })

  gan_lib.DATASETS.update(dataset.get_datasets())
  params.DATASET_PARAMS.update(dataset.get_dataset_params())
