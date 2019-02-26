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

"""Abstract class that describes a single evaluation task.

The tasks can be run in or after session. Each task can result
in a set of metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from absl import flags
import six
import tensorflow as tf

FLAGS = flags.FLAGS


@six.add_metaclass(abc.ABCMeta)
class EvalTask(object):
  """Class that describes a single evaluation task.

  For example: compute inception score or compute accuracy.
  The classes that inherit from it, should implement the methods below.
  """

  _LABEL = None

  def metric_list(self):
    """List of metrics that this class generates.

    These are the only keys that RunXX methods can return in
    their output maps.
    Returns:
      frozenset of strings, which are the names of the metrics that task
      computes.
    """
    return frozenset(self._LABEL)

  def _create_session(self):
    try:
      target = FLAGS.master
    except AttributeError:
      return tf.Session()
    return tf.Session(target)

  @abc.abstractmethod
  def run_after_session(self, fake_dset, real_dset):
    """Runs the task after all the generator calls, after session was closed.

    WARNING: the images here, are in 0..255 range, with 3 color channels.

    Args:
      fake_dset: `EvalDataSample` with fake images and inception features.
      real_dset: `EvalDataSample` with real images and inception features.

    Returns:
      Dict with metric values. The keys must be contained in the set that
      "MetricList" method above returns.
    """
