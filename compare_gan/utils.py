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

"""Utilities library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect

from absl import logging
import six


# In Python 2 the inspect module does not have FullArgSpec. Define a named tuple
# instead.
if hasattr(inspect, "FullArgSpec"):
  _FullArgSpec = inspect.FullArgSpec  # pylint: disable=invalid-name
else:
  _FullArgSpec = collections.namedtuple("FullArgSpec", [
      "args", "varargs", "varkw", "defaults", "kwonlyargs", "kwonlydefaults",
      "annotations"
  ])


def _getfullargspec(fn):
  """Python 2/3 compatible version of the inspect.getfullargspec method.

  Args:
    fn: The function object.

  Returns:
    A FullArgSpec. For Python 2 this is emulated by a named tuple.
  """
  arg_spec_fn = inspect.getfullargspec if six.PY3 else inspect.getargspec
  try:
    arg_spec = arg_spec_fn(fn)
  except TypeError:
    # `fn` might be a callable object.
    arg_spec = arg_spec_fn(fn.__call__)
  if six.PY3:
    assert isinstance(arg_spec, _FullArgSpec)
    return arg_spec
  return _FullArgSpec(
      args=arg_spec.args,
      varargs=arg_spec.varargs,
      varkw=arg_spec.keywords,
      defaults=arg_spec.defaults,
      kwonlyargs=[],
      kwonlydefaults=None,
      annotations={})


def _has_arg(fn, arg_name):
  """Returns True if `arg_name` might be a valid parameter for `fn`.

  Specifically, this means that `fn` either has a parameter named
  `arg_name`, or has a `**kwargs` parameter.

  Args:
    fn: The function to check.
    arg_name: The name fo the parameter.

  Returns:
    Whether `arg_name` might be a valid argument of `fn`.
  """
  while isinstance(fn, functools.partial):
    fn = fn.func
  while hasattr(fn, "__wrapped__"):
    fn = fn.__wrapped__
  arg_spec = _getfullargspec(fn)
  if arg_spec.varkw:
    return True
  return arg_name in arg_spec.args or arg_name in arg_spec.kwonlyargs


def call_with_accepted_args(fn, **kwargs):
  """Calls `fn` only with the keyword arguments that `fn` accepts."""
  kwargs = {k: v for k, v in six.iteritems(kwargs) if _has_arg(fn, k)}
  logging.debug("Calling %s with args %s.", fn, kwargs)
  return fn(**kwargs)


def get_parameter_overview(variables, limit=40):
  """Returns a string with variables names, their shapes, count, and types.

  To get all trainable parameters pass in `tf.trainable_variables()`.

  Args:
    variables: List of `tf.Variable`(s).
    limit: If not `None`, the maximum number of variables to include.

  Returns:
    A string with a table like in the example.

  +----------------+---------------+------------+---------+
  | Name           | Shape         | Size       | Type    |
  +----------------+---------------+------------+---------+
  | FC_1/weights:0 | (63612, 1024) | 65,138,688 | float32 |
  | FC_1/biases:0  |       (1024,) |      1,024 | float32 |
  | FC_2/weights:0 |    (1024, 32) |     32,768 | float32 |
  | FC_2/biases:0  |         (32,) |         32 | float32 |
  +----------------+---------------+------------+---------+

  Total: 65,172,512
  """
  max_name_len = max([len(v.name) for v in variables] + [len("Name")])
  max_shape_len = max([len(str(v.get_shape())) for v in variables] + [len(
      "Shape")])
  max_size_len = max([len("{:,}".format(v.get_shape().num_elements()))
                      for v in variables] + [len("Size")])
  max_type_len = max([len(v.dtype.base_dtype.name) for v in variables] + [len(
      "Type")])

  var_line_format = "| {: <{}s} | {: >{}s} | {: >{}s} | {: <{}s} |"
  sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")

  header = var_line_format.replace(">", "<").format("Name", max_name_len,
                                                    "Shape", max_shape_len,
                                                    "Size", max_size_len,
                                                    "Type", max_type_len)
  separator = sep_line_format.format("", max_name_len, "", max_shape_len, "",
                                     max_size_len, "", max_type_len)

  lines = [separator, header, separator]

  total_weights = sum(v.get_shape().num_elements() for v in variables)

  # Create lines for up to 80 variables.
  for v in variables:
    if limit is not None and len(lines) >= limit:
      lines.append("[...]")
      break
    lines.append(var_line_format.format(
        v.name, max_name_len,
        str(v.get_shape()), max_shape_len,
        "{:,}".format(v.get_shape().num_elements()), max_size_len,
        v.dtype.base_dtype.name, max_type_len))

  lines.append(separator)
  lines.append("Total: {:,}".format(total_weights))

  return "\n".join(lines)


def log_parameter_overview(variables, msg):
  """Writes a table with variables name and shapes to INFO log.

    See get_parameter_overview for details.

  Args:
    variables: List of `tf.Variable`(s).
    msg: Message to be logged before the table.
  """
  table = get_parameter_overview(variables, limit=None)
  # The table can to large to fit into one log entry.
  lines = [msg] + table.split("\n")
  for i in range(0, len(lines), 80):
    logging.info("\n%s", "\n".join(lines[i:i + 80]))
