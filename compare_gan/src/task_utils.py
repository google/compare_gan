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

"""Functions for creating list of tasks and writing them to disk."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os

from compare_gan.src import simple_task_pb2

import six
import tensorflow as tf
from google.protobuf import text_format


def ParseOptions(task):
  """Parse the options from a task proto into a dictionary."""
  options = {}
  for dim in task.dimensions:
    v = None
    if dim.HasField("bool_value"):
      v = dim.bool_value
    elif dim.HasField("string_value"):
      v = dim.string_value
    elif dim.HasField("int_value"):
      v = dim.int_value
    elif dim.HasField("float_value"):
      v = dim.float_value
    assert v is not None
    options[dim.parameter] = v

  return options


def UnrollCalls(function, kwargs):
  """Call function with all arguments unrolled and concatenate results.

  Unroll all list arguments in kwargs as multiple calls to function, and
  concatenate the results. This enables things like cross-product.

  Args:
    function: the function to execute
    kwargs: a dictionary specifying which arguments to use (as explained
      below)

  Returns:
    a list whose elements are the results of the calls.

  Example usage:
    kwargs = {"a": 1, "b": [2, 4], "c": [5, 6], "d": 7, "e": [8]}
    will call "function" with a=1, d=7, e=8 and all combinations of
    b x c in [2,4] x [5, 6].
  """
  res = []
  for key, value in sorted(six.iteritems(kwargs)):
    assert not isinstance(key, tuple)
    if isinstance(value, list):
      for v in value:
        kwargs_copy = copy.deepcopy(kwargs)
        kwargs_copy[key] = v
        res += UnrollCalls(function, kwargs_copy)
      return res
  res.append(function(**kwargs))
  return res


def CrossProduct(config):
  """Computes the cross product of all options in config.

  Following common.UnrollCalls interpretation of a dict containing options,
  this compute the cross product and returns a list of those options from the
  cross product.

  Args:
    config: a dictionary, following common.UnrollCalls format.

  Returns:
    A list of dictionaries, containing the cross product of all options.
  """
  def _Copy(**args):
    return copy.deepcopy(args)
  options = UnrollCalls(_Copy, config)
  return options


def MakeDimensions(dim_dict, extra_dims=None, base_task=None):
  """Creates a task proto from given dictionary."""
  task = simple_task_pb2.Task()
  if base_task is not None:
    task.MergeFrom(base_task)
  if extra_dims is not None:
    dim_dict = copy.copy(dim_dict)
    dim_dict.update(extra_dims)
  dim_dict = collections.OrderedDict(sorted(dim_dict.items()))
  for key, value in six.iteritems(dim_dict):
    if key in ("_proto", "_prefix"):
      # We skip the special keys.
      continue
    dimension = task.dimensions.add()
    dimension.parameter = key
    if isinstance(value, str):
      dimension.string_value = value
    elif isinstance(value, bool):
      dimension.bool_value = value
    elif isinstance(value, int):
      dimension.int_value = value
    elif isinstance(value, float):
      dimension.float_value = value
    elif isinstance(value, tuple):
      # Convert the tuple in a string
      dimension.string_value = str(value)
    else:
      # We skip the values that are of other type,
      # e.g. proto (this is the case of the "grid" argument that is
      # used in many functions).
      del task.dimensions[-1]
      continue
  return task


def WriteTasksToDirectories(dirname, options):
  print ("Writing tasks to directory: %s" % dirname)
  if not tf.gfile.Exists(dirname):
    tf.gfile.MakeDirs(dirname)

  for i, opt in enumerate(options):
    task = MakeDimensions(opt)
    task.num = i
    tf.gfile.MkDir(os.path.join(dirname, str(i)))
    with tf.gfile.Open(os.path.join(dirname, str(i), "task"), "w") as f:
      f.write(text_format.MessageToString(task))
