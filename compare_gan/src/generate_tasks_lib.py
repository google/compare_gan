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

"""Generate tasks for comparing GANs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import csv
import math
import os
import random

from compare_gan.src import params
from compare_gan.src import simple_task_pb2
from compare_gan.src import task_utils

from compare_gan.src.gans import consts
import six
import tensorflow as tf
from google.protobuf import text_format

flags = tf.flags
FLAGS = flags.FLAGS
logging = tf.logging

BATCH_SIZE = 64

ALL_GANS = [
    "GAN", "GAN_MINMAX", "WGAN", "WGAN_GP",
    "DRAGAN", "VAE", "LSGAN", "BEGAN"]


def CreateCrossProductAndAddDefaultParams(config):
  tasks = task_utils.CrossProduct(config)
  # Add GAN and dataset specific hyperparams.
  for task in tasks:
    defaults = GetDefaultParams(
        params.GetParameters(task["gan_type"], "wide"))
    defaults.update(task)
    task.update(defaults)
  return tasks


def TestExp():
  """Run for one epoch over all tested GANs."""
  config = {
      "gan_type": ["GAN", "GAN_MINMAX", "WGAN", "WGAN_GP",
                   "DRAGAN", "VAE", "LSGAN", "BEGAN",
                   "GAN_PENALTY", "WGAN_PENALTY"],
      "dataset": ["fake"],
      "architecture": [consts.INFOGAN_ARCH],
      "training_steps": [100],
      # Don't save any checkpoints during the training
      # (one is always saved at the end).
      "save_checkpoint_steps": [10000],
      "batch_size": [BATCH_SIZE],
      "tf_seed": [42],
      "z_dim": [64],
      # For test reasons - use a small sample.
      "eval_test_samples": [50],
  }
  return CreateCrossProductAndAddDefaultParams(config)


def TestGILBOExp():
  config = TestExp()[0]
  config["compute_gilbo"] = True
  config["gilbo_max_train_cycles"] = 2
  config["gilbo_train_steps_per_cycle"] = 100
  config["gilbo_eval_steps"] = 100
  return [config]


def TestGansWithPenalty():
  """Run for one epoch over all tested GANs."""
  config = {
      "gan_type": ["GAN_PENALTY", "WGAN_PENALTY"],
      "penalty_type": [consts.NO_PENALTY, consts.WGANGP_PENALTY],
      "dataset": ["mnist"],
      "training_steps": [60000 // BATCH_SIZE],
      "save_checkpoint_steps": [10000],
      "architecture": [consts.INFOGAN_ARCH],
      "batch_size": [BATCH_SIZE],
      "tf_seed": [42],
      "z_dim": [64],
  }
  return CreateCrossProductAndAddDefaultParams(config)


def TestNormalization():
  """Run for one epoch over different normalizations."""
  config = {
      "gan_type": ["GAN_PENALTY", "WGAN_PENALTY"],
      "penalty_type": [consts.NO_PENALTY],
      "discriminator_normalization": [consts.BATCH_NORM,
                                      consts.SPECTRAL_NORM],
      "dataset": ["mnist"],
      "training_steps": [60000 // BATCH_SIZE],
      "save_checkpoint_steps": [10000],
      "architecture": [consts.INFOGAN_ARCH],
      "batch_size": [BATCH_SIZE],
      "tf_seed": [42],
      "z_dim": [64],
  }
  return CreateCrossProductAndAddDefaultParams(config)


def TestNewDatasets():
  """Run for one epoch over all tested GANs."""
  config = {
      "gan_type": ["GAN_PENALTY"],
      "penalty_type": [consts.NO_PENALTY],
      "dataset": ["celebahq128", "imagenet64", "imagenet128", "lsun-bedroom"],
      "training_steps": [100],
      "save_checkpoint_steps": [50],
      "architecture": [consts.DCGAN_ARCH],
      "batch_size": [BATCH_SIZE],
      "tf_seed": [42],
      "z_dim": [100],
  }
  return CreateCrossProductAndAddDefaultParams(config)


def TestGansWithPenaltyNewDatasets(architecture):
  """Run for one epoch over all tested GANs."""
  config = {
      "gan_type": ["GAN_PENALTY", "WGAN_PENALTY"],
      "penalty_type": [consts.NO_PENALTY, consts.WGANGP_PENALTY],
      "dataset": ["celebahq128", "imagenet128", "lsun-bedroom"],
      "training_steps": [162000 * 80 // BATCH_SIZE],
      "save_checkpoint_steps": [10000],
      "architecture": [architecture],
      "batch_size": [BATCH_SIZE],
      "tf_seed": [42],
      "z_dim": [128],
  }
  return CreateCrossProductAndAddDefaultParams(config)


def GetDefaultParams(gan_params):
  """Return the default params for a GAN (=the ones used in the paper)."""
  ret = {}
  for param_name, param_info in six.iteritems(gan_params):
    ret[param_name] = param_info.default
  return ret


def BestModelSNDCGan():
  # These models are matching table 5 (SNDCGan) from the paper.
  best_models = [
    ## Without normalization and without penalty.
    # Line 1: FID score: 28.66
    {
      "dataset": "cifar10",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0001,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 2: FID score: 33.23
    {
      "dataset": "cifar10",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 3: FID score: 61.15
    {
      "dataset": "celebahq128",
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "penalty_type": consts.NO_PENALTY,
      "tf_seed": 23,
      "training_steps": 100000,
      "learning_rate": 0.000412,
      "beta1": 0.246,
      "beta2": 0.341,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 4: FID score: 62.96
    {
      "dataset": "celebahq128",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "learning_rate": 0.000254,
      "beta1": 0.222,
      "beta2": 0.599,
      "disc_iters": 1,
      "tf_seed": 23,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 5: FID score: 163.51
    {
      "dataset": "lsun-bedroom",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 23,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 6: FID score: 167.15
    {
      "dataset": "lsun-bedroom",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.000062,
      "beta1": 0.608,
      "beta2": 0.620,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 23,
      "lambda": 10, # this should not be needed for this one.
    },

    ## With normalization

    # Line 7: FID score: 25.27
    {
      "dataset": "cifar10",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },
    # Line 8: FID score: 26.16
    {
      "dataset": "cifar10",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0001,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 10,
    },
    # Line 9: FID score: 28.21
    {
      "dataset": "celebahq128",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.9,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 10,
    },
    # Line 10: FID score: 29.92
    {
      "dataset": "celebahq128",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.000154,
      "beta1": 0.246,
      "beta2": 0.734,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 0.144,
    },
    # Line 11: FID score: 53.59
    {
      "dataset": "lsun-bedroom",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.000264,
      "beta1": 0.011,
      "beta2": 0.966,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 0.300,
    },
    # Line 12: FID score: 56.71
    {
      "dataset": "lsun-bedroom",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.00192,
      "beta1": 0.097,
      "beta2": 0.938,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 0.580,
    },

    ## With normalization and penalty

    # Line 13: FID score: 25.27
    {
      "dataset": "cifar10",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },
    # Line 14: FID score: 26.00
    {
      "dataset": "cifar10",
      "training_steps": 200000,
      "penalty_type": consts.WGANGP_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 10,
    },
    # Line 15: FID score: 24.67
    {
      "dataset": "celebahq128",
      "training_steps": 100000,
      "penalty_type": consts.WGANGP_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },
    # Line 16: FID score: 25.22
    {
      "dataset": "celebahq128",
      "training_steps": 100000,
      "penalty_type": consts.WGANGP_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.900,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },
    # Line 17: FID score: 53.59
    {
      "dataset": "lsun-bedroom",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.000264,
      "beta1": 0.011,
      "beta2": 0.966,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 0.300,
    },
    # Line 18: FID score: 56.71
    {
      "dataset": "lsun-bedroom",
      "training_steps": 100000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.00192,
      "beta1": 0.097,
      "beta2": 0.938,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 0.580,
    },

  ]

  for model in best_models:
    model.update({
      "architecture": consts.SNDCGAN_ARCH,
      "batch_size": 64,
      "gan_type": consts.GAN_WITH_PENALTY,
      "optimizer": "adam",
      "save_checkpoint_steps": 20000,
      "z_dim": 128,
    })

  return best_models


def BestModelResnet19():
  # These models are matching table 7 (ResNet19) from the paper.
  best_models = [
    ## Without normalization and without penalty.
    # Line 1: FID score: 34.29
    {
      "dataset": "celebahq128",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0000338,
      "beta1": 0.3753,
      "beta2": 0.9982,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 2: FID score: 35.85
    {
      "dataset": "celebahq128",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0001,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 3: FID score: 102.74
    {
      "dataset": "lsun-bedroom",
      "gan_type": consts.LSGAN_WITH_PENALTY,
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0000322,
      "beta1": 0.5850,
      "beta2": 0.9904,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 4: FID score: 112.92
    {
      "dataset": "lsun-bedroom",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0000193,
      "beta1": 0.1947,
      "beta2": 0.8819,
      "disc_iters": 1,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },

    ## With normalization

    # Line 5: FID score: 30.02
    {
      "dataset": "celebahq128",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0001,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.LAYER_NORM,
      "tf_seed": 2,
      "lambda": 0.001,
    },
    # Line 6: FID score: 32.05
    {
      "dataset": "celebahq128",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0001,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.LAYER_NORM,
      "tf_seed": 2,
      "lambda": 0.01,
    },
    # Line 7: FID score: 41.6
    {
      "dataset": "lsun-bedroom",
      "gan_type": consts.LSGAN_WITH_PENALTY,
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },
    # Line 8: FID score: 42.51
    {
      "dataset": "lsun-bedroom",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002851,
      "beta1": 0.1019,
      "beta2": 0.998,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 5.6496,
    },

    ## With normalization and penalty

    # Line 9: FID score: 29.04
    {
      "dataset": "celebahq128",
      "training_steps": 200000,
      "penalty_type": consts.WGANGP_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },
    # Line 10: FID score: 29.13
    {
      "dataset": "celebahq128",
      "training_steps": 200000,
      "penalty_type": consts.DRAGAN_PENALTY,
      "learning_rate": 0.0001,
      "beta1": 0.5,
      "beta2": 0.9,
      "disc_iters": 5,
      "discriminator_normalization": consts.LAYER_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },
    # Line 11: FID score: 40.36
    {
      "dataset": "lsun-bedroom",
      "training_steps": 200000,
      "penalty_type": consts.WGANGP_PENALTY,
      "learning_rate": 0.0001281,
      "beta1": 0.7108,
      "beta2": 0.9792,
      "disc_iters": 1,
      "discriminator_normalization": consts.LAYER_NORM,
      "tf_seed": 2,
      "lambda": 0.1451,
    },
    # Line 12: FID score: 41.60
    {
      "dataset": "lsun-bedroom",
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 1,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    }]

  for model in best_models:
    model.update({
      "architecture": consts.RESNET5_ARCH,
      "batch_size": 64,
      "optimizer": "adam",
      "save_checkpoint_steps": 20000,
      "z_dim": 128,
    })
    if "gan_type" not in model:
      model.update({"gan_type": consts.GAN_WITH_PENALTY})

  return best_models

def BestModelResnetCifar():
  # These models are matching table 7 (ResNet19) from the paper.
  best_models = [
    ## Without normalization and without penalty.
    # Line 1: FID score: 28.12
    {
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 5,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },
    # Line 2: FID score: 30.08
    {
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0001,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 5,
      "discriminator_normalization": consts.NO_NORMALIZATION,
      "tf_seed": 2,
      "lambda": 10, # this should not be needed for this one.
    },

    ## With normalization.

    # Line 3: FID score: 22.91
    {
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 5,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 10,
    },

    # Line 4: FID score: 23.22
    {
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 5,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },

    ## With normalization and penalty.

    # Line 5: FID score: 22.73
    {
      "training_steps": 200000,
      "penalty_type": consts.WGANGP_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 5,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 1,
    },

    # Line 6: FID score: 22.91
    {
      "training_steps": 200000,
      "penalty_type": consts.NO_PENALTY,
      "learning_rate": 0.0002,
      "beta1": 0.5,
      "beta2": 0.999,
      "disc_iters": 5,
      "discriminator_normalization": consts.SPECTRAL_NORM,
      "tf_seed": 2,
      "lambda": 10,
    },
  ]

  for model in best_models:
    model.update({
      "architecture": consts.RESNET_CIFAR,
      "dataset": "cifar10",
      "batch_size": 64,
      "gan_type": consts.GAN_WITH_PENALTY,
      "optimizer": "adam",
      "save_checkpoint_steps": 20000,
      "z_dim": 128,
    })

  return best_models


def GetTasks(experiment):
  """Get a list of tasks to run and eval for the given experiment name."""
  random.seed(123)

  if experiment == "test":
    return TestExp()
  if experiment == "test_gilbo":
    return TestGILBOExp()
  elif experiment == "test_penalty":
    return TestGansWithPenalty()
  elif experiment == "test_new_datasets":
    return TestNewDatasets()
  elif experiment == "test_penalty_new_datasets":
    return TestGansWithPenaltyNewDatasets(consts.DCGAN_ARCH)
  elif experiment == "test_penalty_resnet5":
    return TestGansWithPenaltyNewDatasets(consts.RESNET5_ARCH)
  elif experiment == "test_normalization":
    return TestNormalization();
  elif experiment == "best_models_sndcgan":
    return BestModelSNDCGan();
  elif experiment == "best_models_resnet19":
    return BestModelResnet19();
  elif experiment == "best_models_resnet_cifar":
    return BestModelResnetCifar();
  else:
    raise ValueError("Unknown experiment %s" % experiment)
