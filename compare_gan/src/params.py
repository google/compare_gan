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


ParamInfo = collections.namedtuple(
    "ParamInfo", ["default", "range", "is_log_scale", "is_discrete"])

NARROW_RANGE = "narrow"
WIDE_RANGE = "wide"


class ParamRanges(object):
  """Class for holding parameter ranges."""

  def __init__(self):
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
  param_ranges.AddRange("discriminator_batchnorm", True, [True, False],
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
    param_ranges.AddRange("discriminator_batchnorm", True, [True, False],
                          is_log_scale=False, is_discrete=True)

  param_ranges.UpdateDefaults(
      {"beta1": 0.0, "learning_rate": 0.00005, "disc_iters": 1,
       "discriminator_batchnorm": True})
  return param_ranges.GetParams()


def LSGANHyperParams(range_type):
  """Returns LSGAN hyperparameters.

  Args:
    range_type: which param defaults to use.
  """
  param_ranges = GetDefaultRange(range_type)
  if range_type == NARROW_RANGE:
    param_ranges.AddRange("discriminator_batchnorm", True, [True, False],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0002, "disc_iters": 1,
       "discriminator_batchnorm": True})
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
    param_ranges.AddRange("discriminator_batchnorm", True, [True, False],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.00005, "disc_iters": 5,
       "discriminator_batchnorm": True})
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
    param_ranges.AddRange("discriminator_batchnorm", False, [False],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0001, "disc_iters": 5,
       "discriminator_batchnorm": False})
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
    param_ranges.AddRange("discriminator_batchnorm", False, [False],
                          is_log_scale=False, is_discrete=True)
  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0001, "disc_iters": 1,
       "discriminator_batchnorm": True})
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
    param_ranges.AddRange("discriminator_batchnorm", False, [True, False],
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
    param_ranges.AddRange("discriminator_batchnorm", False, [True, False],
                          is_log_scale=False, is_discrete=True)

  param_ranges.UpdateDefaults(
      {"beta1": 0.5, "learning_rate": 0.0001, "disc_iters": 5,
       "discriminator_batchnorm": False})
  param_ranges.AddRange("lambda", 0.001, lambda_range,
                        is_log_scale=True, is_discrete=False)
  param_ranges.AddRange("gamma", 0.75, gamma_range,
                        is_log_scale=False, is_discrete=False)
  return param_ranges.GetParams()


def GetParameters(gan_type, unused_dataset_name, range_type="narrow"):
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
  else:
    raise NotImplementedError("Unknown GAN: %s" % gan_type)


def GetDatasetParameters(dataset_name):
  """Returns default dataset parameters for a specific dataset."""

  if dataset_name in ["mnist", "fashion-mnist", "triangles", "squares"]:
    return {
        "input_height": 28,
        "input_width": 28,
        "output_height": 28,
        "output_width": 28,
        "c_dim": 1,
        "dataset_name": dataset_name
    }
  elif dataset_name in ["cifar10"]:
    return {
        "input_height": 32,
        "input_width": 32,
        "output_height": 32,
        "output_width": 32,
        "c_dim": 3,
        "dataset_name": dataset_name
    }
  elif dataset_name in ["celeba", "celeba20k"]:
    return {
        "input_height": 64,
        "input_width": 64,
        "output_height": 64,
        "output_width": 64,
        "c_dim": 3,
        "dataset_name": dataset_name
    }
  else:
    raise NotImplementedError("Parameters not defined for '%s'" % dataset_name)
