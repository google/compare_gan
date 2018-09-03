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

"""Hyperparameter ranges for various GANs.

We define the default GAN parameters with respect to the datasets and the
training hyperparameters. The hyperparameters used by the respective authors
are also added to the set.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from compare_gan.src.gans import consts


ParamInfo = collections.namedtuple(
    "ParamInfo", ["default", "range", "is_log_scale", "is_discrete"])

NARROW_RANGE = "narrow"
WIDE_RANGE = "wide"


class ParamRanges(object):
  """Class for holding parameter ranges."""

  def __init__(self):
    super(ParamRanges, self).__init__()
    self._params = {}

  def AddRange(self, param_name, default, value_range,
               is_log_scale, is_discrete):
    self._params[param_name] = ParamInfo(
        default, value_range, is_log_scale, is_discrete)

  def _UpdateDefault(self, param_name, default_value):
    old_param_info = self._params[param_name]
    new_param_info = ParamInfo(
        default_value, old_param_info.range,
        old_param_info.is_log_scale, old_param_info.is_discrete)
    self._params[param_name] = new_param_info

  def UpdateDefaults(self, params_to_update):
    for k, v in params_to_update.items():
      self._UpdateDefault(k, v)

  def GetParams(self):
    return self._params


def GetDefaultWideRange():
  """Get default ranges for tuning (wide, before pruning)."""
  param_ranges = ParamRanges()
  param_ranges.AddRange("beta1", 0.5, [0.0, 1.0],
                        is_log_scale=False, is_discrete=False)
  param_ranges.AddRange("learning_rate", 0.0001, [-5.0, -2.0],
                        is_log_scale=True, is_discrete=False)
  param_ranges.AddRange("disc_iters", 1, [1, 5],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("discriminator_normalization", consts.BATCH_NORM,
                        consts.NORMALIZERS,
                        is_log_scale=False, is_discrete=True)
  return param_ranges


def GetDefaultNarrowRange():
  """Get default ranges for tuning (narrow, after pruning)."""
  param_ranges = ParamRanges()
  param_ranges.AddRange("beta1", 0.5, [0.5],
                        is_log_scale=False, is_discrete=True)
  param_ranges.AddRange("learning_rate", 0.0001, [-4.0, -3.0],
                        is_log_scale=True, is_discrete=False)
  param_ranges.AddRange("disc_iters", 1, [1],
                        is_log_scale=False, is_discrete=True)
  # NOTE: batchnorm really depends on the model, so we will specify this
  # range in each class separately.
  return param_ranges


def GetDefaultRange(range_type):
  if range_type == WIDE_RANGE:
    return GetDefaultWideRange()
  elif range_type == NARROW_RANGE:
    return GetDefaultNarrowRange()
  else:
    assert False, "Unsupported range: %s" % range_type


def GANHyperParams(range_type):
  """Returns GAN hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)
  if range_type == NARROW_RANGE:
    param_ranges.AddRange("discriminator_normalization",
                          consts.BATCH_NORM, [consts.BATCH_NORM,
                                              consts.NO_NORMALIZATION],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.0, "learning_rate": 0.00005, "disc_iters": 1,
       "discriminator_normalization": consts.BATCH_NORM})
  return param_ranges.GetParams()


def LSGANHyperParams(range_type):
  """Returns LSGAN hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)
  if range_type == NARROW_RANGE:
    param_ranges.AddRange("discriminator_normalization",
                          consts.BATCH_NORM, [consts.BATCH_NORM,
                                              consts.NO_NORMALIZATION],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0002, "disc_iters": 1,
       "discriminator_normalization": consts.BATCH_NORM})
  return param_ranges.GetParams()


def WGANHyperParams(range_type):
  """Returns WGAN hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)
  if range_type == WIDE_RANGE:
    weight_range = [-3.0, 0.0]
  elif range_type == NARROW_RANGE:
    weight_range = [-2.0, 0.0]
    param_ranges.AddRange("discriminator_normalization",
                          consts.BATCH_NORM, [consts.BATCH_NORM,
                                              consts.NO_NORMALIZATION],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.00005, "disc_iters": 5,
       "discriminator_normalization": consts.BATCH_NORM})
  param_ranges.AddRange("weight_clipping", 0.01, weight_range,
                        is_log_scale=True, is_discrete=False)
  return param_ranges.GetParams()


def WGANGPHyperParams(range_type):
  """Returns WGAN_GP hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)
  if range_type == WIDE_RANGE:
    lambda_range = [-1.0, 2.0]
  elif range_type == NARROW_RANGE:
    lambda_range = [-1.0, 1.0]
    # From the barplot, clearly False is better.
    param_ranges.AddRange("discriminator_normalization",
                          consts.NO_NORMALIZATION,
                          [consts.NO_NORMALIZATION],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0001, "disc_iters": 5,
       "discriminator_normalization": consts.NO_NORMALIZATION})
  param_ranges.AddRange("lambda", 10.0, lambda_range,
                        is_log_scale=True, is_discrete=False)
  return param_ranges.GetParams()


def DRAGANHyperParams(range_type):
  """Returns DRAGAN hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)
  if range_type == WIDE_RANGE:
    lambda_range = [-1.0, 2.0]
  elif range_type == NARROW_RANGE:
    lambda_range = [-1.0, 1.0]
    # From the barplot, clearly False is better.
    param_ranges.AddRange("discriminator_normalization",
                          consts.NO_NORMALIZATION,
                          [consts.NO_NORMALIZATION],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0001, "disc_iters": 1,
       "discriminator_normalization": consts.BATCH_NORM})
  param_ranges.AddRange("lambda", 10.0, lambda_range,
                        is_log_scale=True, is_discrete=False)
  return param_ranges.GetParams()


def VAEHyperParams(range_type):
  """Returns VAE hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)
  if range_type == NARROW_RANGE:
    # Needed because AbstractGAN expects this param.
    param_ranges.AddRange("discriminator_normalization",
                          consts.NO_NORMALIZATION, [consts.BATCH_NORM,
                                                    consts.NO_NORMALIZATION],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.00005})
  return param_ranges.GetParams()


def BEGANHyperParams(range_type):
  """Returns BEGAN hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)

  if range_type == WIDE_RANGE:
    lambda_range = [-4.0, -2.0]
    gamma_range = [0.0, 1.0]
  elif range_type == NARROW_RANGE:
    lambda_range = [-3.0, -3.0]
    gamma_range = [0.6, 0.9]
    param_ranges.AddRange("discriminator_normalization",
                          consts.NO_NORMALIZATION, [consts.BATCH_NORM,
                                                    consts.NO_NORMALIZATION],
                          is_log_scale=False, is_discrete=True)

  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0001, "disc_iters": 5,
       "discriminator_normalization": consts.NO_NORMALIZATION})
  param_ranges.AddRange("lambda", 0.001, lambda_range,
                        is_log_scale=True, is_discrete=False)
  param_ranges.AddRange("gamma", 0.75, gamma_range,
                        is_log_scale=False, is_discrete=False)
  return param_ranges.GetParams()


def GANPENALTYHyperParams(range_type, gan_type, penalty_type=None):
  """Returns WGAN_GP hyperparameters.

  Args:
    range_type: which param defaults to use.
    gan_type: which of the GANs with penalty we should use
    penalty_type: which of the penalty we should use. If not specified,
                  a default range will be generated.
  """
  param_ranges = GetDefaultRange(range_type)
  param_ranges.AddRange("penalty_type", consts.NO_PENALTY,
                        [consts.NO_PENALTY, consts.WGANGP_PENALTY],
                        is_log_scale=False, is_discrete=True)
  if penalty_type and penalty_type == consts.L2_PENALTY:
    param_ranges.AddRange("lambda", 0.01, [-4.0, 1.0],
                          is_log_scale=True, is_discrete=False)
  else:
    param_ranges.AddRange("lambda", 10.0, [-1.0, 2.0],
                          is_log_scale=True, is_discrete=False)
  param_ranges.AddRange("beta2", 0.999, [0, 1],
                        is_log_scale=False, is_discrete=False)

  if gan_type == "GAN_PENALTY":
    param_ranges.UpdateDefaults(
        {"beta1": 0.0, "learning_rate": 0.00005, "disc_iters": 1,
         "discriminator_normalization": consts.BATCH_NORM})
  elif gan_type == "WGAN_PENALTY":
    param_ranges.UpdateDefaults(
        {"beta1": 0.5, "learning_rate": 0.0001, "disc_iters": 5,
         "discriminator_normalization": consts.NO_NORMALIZATION})
  elif gan_type == "SN_GAN" or gan_type == "LSGAN_PENALTY":
    pass
  else:
    assert False, ("GAN type not recognized: %s." % gan_type)

  return param_ranges.GetParams()

PARAMETERS = {
}


def GetParameters(gan_type, range_type="narrow", penalty_type=None):
  """Returns a tuple of dataset specific parameters and hyperparameters."""
  if gan_type == "GAN" or gan_type == "GAN_MINMAX":
    return GANHyperParams(range_type)
  elif gan_type == "WGAN":
    return WGANHyperParams(range_type)
  elif gan_type == "WGAN_GP":
    return WGANGPHyperParams(range_type)
  elif gan_type == "DRAGAN":
    return DRAGANHyperParams(range_type)
  elif gan_type == "VAE":
    return VAEHyperParams(range_type)
  elif gan_type == "LSGAN":
    return LSGANHyperParams(range_type)
  elif gan_type == "BEGAN":
    return BEGANHyperParams(range_type)
  elif gan_type in ["GAN_PENALTY", "WGAN_PENALTY", "LSGAN_PENALTY", "SN_GAN"]:
    return GANPENALTYHyperParams(range_type, gan_type, penalty_type)
  elif gan_type in PARAMETERS:
    return PARAMETERS[gan_type](range_type, gan_type, penalty_type)
  else:
    raise NotImplementedError("Unknown GAN: %s" % gan_type)


DATASET_PARAMS = {
    "mnist": {
        "input_height": 28,
        "input_width": 28,
        "output_height": 28,
        "output_width": 28,
        "c_dim": 1,
        "dataset_name": "mnist",
        "eval_test_samples": 10000
    },
    "fashion-mnist": {
        "input_height": 28,
        "input_width": 28,
        "output_height": 28,
        "output_width": 28,
        "c_dim": 1,
        "dataset_name": "fashion-mnist",
        "eval_test_samples": 10000
    },
    "triangles": {
        "input_height": 28,
        "input_width": 28,
        "output_height": 28,
        "output_width": 28,
        "c_dim": 1,
        "dataset_name": "triangles",
        "eval_test_samples": 10000
    },
    "squares": {
        "input_height": 28,
        "input_width": 28,
        "output_height": 28,
        "output_width": 28,
        "c_dim": 1,
        "dataset_name": "squares",
        "eval_test_samples": 10000
    },
    "cifar10": {
        "input_height": 32,
        "input_width": 32,
        "output_height": 32,
        "output_width": 32,
        "c_dim": 3,
        "dataset_name": "cifar10",
        "eval_test_samples": 10000
    },
    "fake": {
        "input_height": 64,
        "input_width": 64,
        "output_height": 64,
        "output_width": 64,
        "c_dim": 1,
        "dataset_name": "fake",
        "eval_test_samples": 100
    },
    "celeba": {
        "input_height": 64,
        "input_width": 64,
        "output_height": 64,
        "output_width": 64,
        "c_dim": 3,
        "dataset_name": "celeba",
        "eval_test_samples": 10000
    },
    "lsun-bedroom": {
        "input_height": 128,
        "input_width": 128,
        "output_height": 128,
        "output_width": 128,
        "c_dim": 3,
        "dataset_name": "lsun-bedroom",
        "eval_test_samples": 10000
    },
    "celebahq128": {
        "input_height": 128,
        "input_width": 128,
        "output_height": 128,
        "output_width": 128,
        "c_dim": 3,
        "dataset_name": "celebahq128",
        "eval_test_samples": 3000
    },
}


def GetDatasetParameters(dataset_name):
  """Returns default dataset parameters for a specific dataset."""
  if dataset_name in DATASET_PARAMS:
    return DATASET_PARAMS[dataset_name]
  else:
    raise NotImplementedError("Parameters not defined for '%s'" % dataset_name)
