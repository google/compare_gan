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

from __future__ import division

import numpy as np
import scipy.misc
import tensorflow as tf


def check_folder(log_dir):
  if not tf.gfile.IsDirectory(log_dir):
    tf.gfile.MakeDirs(log_dir)
  return log_dir


def save_images(images, image_path):
  with tf.gfile.Open(image_path, "wb") as f:
    scipy.misc.imsave(f, images * 255.0)


def gaussian(batch_size, n_dim, mean=0., var=1.):
  return np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def linear(input_,
           output_size,
           scope=None,
           stddev=0.02,
           bias_start=0.0):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", [shape[1], output_size],
        tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable(
        "bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return tf.matmul(input_, matrix) + bias


def batch_norm(input_, is_training, scope):
  return tf.contrib.layers.batch_norm(
      input_,
      decay=0.999,
      epsilon=0.001,
      updates_collections=None,
      scale=True,
      fused=False,
      is_training=is_training,
      scope=scope)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
    biases = tf.get_variable(
        "biases", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


def deconv2d(
    input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(
        input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    biases = tf.get_variable(
        "biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
