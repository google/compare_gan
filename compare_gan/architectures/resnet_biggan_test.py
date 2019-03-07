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

"""Test whether resnet_biggan matches the original BigGAN architecture.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from compare_gan import utils
from compare_gan.architectures import arch_ops
from compare_gan.architectures import resnet_biggan
import gin
import numpy as np
import tensorflow as tf


def guess_initializer(var, graph=None):
  """Helper function to guess the initializer of a variable.

  The function looks at the operations in the initializer name space for the
  variable (e.g. my_scope/my_var_name/Initializer/*). The TF core initializers
  have characteristic sets of operations that can be used to determine the
  initializer.

  Args:
    var: `tf.Variable`. The function will use the name to look for initializer
      operations in the same scope.
    graph: Optional `tf.Graph` that contains the variable. If None the default
      graph is used.

  Returns:
    Tuple of the name of the guessed initializer.
  """
  if graph is None:
    graph = tf.get_default_graph()
  prefix = var.op.name + "/Initializer"
  ops = [op for op in graph.get_operations()
         if op.name.startswith(prefix)]
  assert ops, "No operations found for prefix {}".format(prefix)
  op_names = [op.name[len(prefix) + 1:] for op in ops]
  if len(op_names) == 1:
    if op_names[0] == "Const":
      value = ops[0].get_attr("value").float_val[0]
      if value == 0.0:
        return "zeros"
      if np.isclose(value, 1.0):
        return "ones"
      return "constant"
    return op_names[0]  # ones or zeros
  if "Qr" in op_names and "DiagPart" in op_names:
    return "orthogonal"
  if "random_uniform" in op_names:
    return "glorot_uniform"
  stddev_ops = [op for op in ops if op.name.endswith("stddev")]
  if stddev_ops:
    assert len(stddev_ops) == 1
    stddev = stddev_ops[0].get_attr("value").float_val[0]
  else:
    stddev = None
  if "random_normal" in op_names:
    return "random_normal"
  if "truncated_normal" in op_names:
    if len(str(stddev)) > 5:
      return "glorot_normal"
    return "truncated_normal"


class ResNet5BigGanTest(tf.test.TestCase):

  def setUp(self):
    super(ResNet5BigGanTest, self).setUp()
    gin.clear_config()

  def testNumberOfParameters(self):
    with tf.Graph().as_default():
      batch_size = 16
      z = tf.zeros((batch_size, 120))
      y = tf.one_hot(tf.ones((batch_size,), dtype=tf.int32), 1000)
      generator = resnet_biggan.Generator(
          image_shape=(128, 128, 3),
          batch_norm_fn=arch_ops.conditional_batch_norm)
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      self.assertEqual(fake_images.shape.as_list(), [batch_size, 128, 128, 3])
      discriminator = resnet_biggan.Discriminator()
      predictions = discriminator(fake_images, y, is_training=True)
      self.assertLen(predictions, 3)

      t_vars = tf.trainable_variables()
      g_vars = [var for var in t_vars if "generator" in var.name]
      d_vars = [var for var in t_vars if "discriminator" in var.name]
      g_param_overview = utils.get_parameter_overview(g_vars, limit=None)
      d_param_overview = utils.get_parameter_overview(d_vars, limit=None)
      logging.info("Generator variables:\n%s", g_param_overview)
      logging.info("Discriminator variables:\n%s", d_param_overview)

      for v in g_vars:
        parts = v.op.name.split("/")
        layer, var_name = parts[-2], parts[-1]
        layers_with_bias = {"fc_noise", "up_conv_shortcut", "up_conv1",
                            "same_conv2", "final_conv"}
        # No biases in conditional BN or self-attention.
        if layer not in layers_with_bias:
          self.assertNotEqual(var_name, "bias", msg=str(v))

        # Batch norm variables.
        if parts[-3] == "condition":
          if parts[-4] == "final_bn":
            self.assertEqual(var_name, "kernel", msg=str(v))
            self.assertEqual(v.shape.as_list(), [1, 1, 1, 96], msg=str(v))
          else:
            self.assertEqual(var_name, "kernel", msg=str(v))
            self.assertEqual(v.shape[0].value, 148, msg=str(v))

        # Embedding layer.
        if layer == "embed_y":
          self.assertEqual(var_name, "kernel", msg=str(v))
          self.assertAllEqual(v.shape.as_list(), [1000, 128], msg=str(v))

        # Shortcut connections use 1x1 convolution.
        if layer == "up_conv_shortcut" and var_name == "kernel":
          self.assertEqual(v.shape.as_list()[:2], [1, 1], msg=str(v))
      g_num_weights = sum([v.get_shape().num_elements() for v in g_vars])
      self.assertEqual(g_num_weights, 70433988)

      for v in d_vars:
        parts = v.op.name.split("/")
        layer, var_name = parts[-2], parts[-1]
        layers_with_bias = {"down_conv_shortcut", "same_conv1", "down_conv2",
                            "same_conv_shortcut", "same_conv2", "final_fc"}
        # No biases in conditional BN or self-attention.
        if layer not in layers_with_bias:
          self.assertNotEqual(var_name, "bias", msg=str(v))

        # no Shortcut in last block.
        if parts[-3] == "B6":
          self.assertNotEqual(layer, "same_shortcut", msg=str(v))
      d_num_weights = sum([v.get_shape().num_elements() for v in d_vars])
      self.assertEqual(d_num_weights, 87982370)

  def testInitializers(self):
    gin.bind_parameter("weights.initializer", "orthogonal")
    with tf.Graph().as_default():
      z = tf.zeros((8, 120))
      y = tf.one_hot(tf.ones((8,), dtype=tf.int32), 1000)
      generator = resnet_biggan.Generator(
          image_shape=(128, 128, 3),
          batch_norm_fn=arch_ops.conditional_batch_norm)
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      discriminator = resnet_biggan.Discriminator()
      discriminator(fake_images, y, is_training=True)

      for v in tf.trainable_variables():
        parts = v.op.name.split("/")
        layer, var_name = parts[-2], parts[-1]
        initializer_name = guess_initializer(v)
        logging.info("%s => %s", v.op.name, initializer_name)
        if layer == "embedding_fc" and var_name == "kernel":
          self.assertEqual(initializer_name, "glorot_normal")
        elif layer == "non_local_block" and var_name == "sigma":
          self.assertEqual(initializer_name, "zeros")
        elif layer == "final_norm" and var_name == "gamma":
          self.assertEqual(initializer_name, "ones")
        elif layer == "final_norm" and var_name == "beta":
          self.assertEqual(initializer_name, "zeros")
        elif var_name == "kernel":
          self.assertEqual(initializer_name, "orthogonal")
        elif var_name == "bias":
          self.assertEqual(initializer_name, "zeros")
        else:
          self.fail("Unknown variables {}".format(v))


if __name__ == "__main__":
  tf.test.main()
