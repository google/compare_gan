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

"""Utility classes and methods for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from absl import flags
from compare_gan import eval_utils
from compare_gan.architectures import abstract_arch
from compare_gan.architectures import arch_ops
import gin
import mock
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


def create_fake_inception_graph():
  """Creates a graph that mocks inception.

  It takes the input, multiplies it through a matrix full of 0.001 values
  and returns as logits. It makes sure to match the tensor names of
  the real inception model.

  Returns:
    tf.Graph object with a simple mock inception inside.
  """
  fake_inception = tf.Graph()
  with fake_inception.as_default():
    graph_input = tf.placeholder(
        tf.float32, shape=[None, 299, 299, 3], name="Mul")
    matrix = tf.ones(shape=[299 * 299 * 3, 10]) * 0.00001
    output = tf.matmul(tf.layers.flatten(graph_input), matrix)
    output = tf.identity(output, name="pool_3")
    output = tf.identity(output, name="logits")
  return fake_inception.as_graph_def()


class Generator(abstract_arch.AbstractGenerator):
  """Generator with a single linear layer from z to the output."""

  def __init__(self, **kwargs):
    super(Generator, self).__init__(**kwargs)
    self.call_arg_list = []

  def apply(self, z, y, is_training):
    self.call_arg_list.append(dict(z=z, y=y, is_training=is_training))
    batch_size = z.shape[0].value
    out = arch_ops.linear(z, np.prod(self._image_shape), scope="fc_noise")
    out = tf.nn.sigmoid(out)
    return tf.reshape(out, [batch_size] + list(self._image_shape))


class Discriminator(abstract_arch.AbstractDiscriminator):
  """Discriminator with a single linear layer."""

  def __init__(self, **kwargs):
    super(Discriminator, self).__init__(**kwargs)
    self.call_arg_list = []

  def apply(self, x, y, is_training):
    self.call_arg_list.append(dict(x=x, y=y, is_training=is_training))
    h = tf.reduce_mean(x, axis=[1, 2])
    out = arch_ops.linear(h, 1)
    return tf.nn.sigmoid(out), out, h


class CompareGanTestCase(tf.test.TestCase):
  """Base class for test cases."""

  def setUp(self):
    super(CompareGanTestCase, self).setUp()
    # Use fake datasets instead of reading real files.
    FLAGS.data_fake_dataset = True
    # Clear the gin cofiguration.
    gin.clear_config()
    # Mock the inception graph.
    fake_inception_graph = create_fake_inception_graph()
    self.inception_graph_def_mock = mock.patch.object(
        eval_utils,
        "get_inception_graph_def",
        return_value=fake_inception_graph).start()

  def _get_empty_model_dir(self):
    unused_sub_dir = str(datetime.datetime.now().microsecond)
    model_dir = os.path.join(FLAGS.test_tmpdir, unused_sub_dir)
    assert not tf.gfile.Exists(model_dir)
    return model_dir
