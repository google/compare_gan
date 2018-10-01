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

"""Provides access to Datasets and their parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("multigan_dataset_root",
                    "/tmp/datasets/multi_gan",
                    "Folder which contains all datasets.")

# Multi-MNIST configs.
MULTI_MNIST_CONFIGS = [
    "multi-mnist-3-uniform", "multi-mnist-3-triplet",
    "multi-mnist-3-uniform-rgb-occluded",
    "multi-mnist-3-uniform-rgb-occluded-cifar10"]


def unpack_clevr_image(image_data):
  """Returns an image and a label. 0-1 range."""
  value = tf.parse_single_example(
      image_data, features={"image": tf.FixedLenFeature([], tf.string)})
  image = tf.decode_raw(value["image"], tf.uint8)
  image = tf.reshape(image, [1, 320, 480, 3])
  image = tf.image.resize_bilinear(image, size=(160, 240))
  image = tf.squeeze(tf.image.resize_image_with_crop_or_pad(image, 128, 128))
  image = tf.cast(image, tf.float32) / 255.0
  dummy_label = tf.constant(value=0, dtype=tf.int32)
  return image, dummy_label


def load_clevr(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  filenames = tf.data.Dataset.list_files(
      os.path.join(FLAGS.multigan_dataset_root, "clevr/%s*" % split_name))

  return tf.data.TFRecordDataset(
      filenames,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack_clevr_image, num_parallel_calls=num_threads)


def unpack_multi_mnist_image(split_name, k, rgb, image_data):
  """Returns an image and a label in [0, 1] range."""
  c_dim = 3 if rgb else 1

  value = tf.parse_single_example(
      image_data,
      features={"%s/image" % split_name: tf.FixedLenFeature([], tf.string),
                "%s/label" % split_name: tf.FixedLenFeature([k], tf.int64)})

  image = tf.decode_raw(value["%s/image" % split_name], tf.float32)
  image = tf.reshape(image, [64, 64, c_dim])
  image = image / 255.0 if rgb else image
  label = tf.cast(value["%s/label" % split_name], tf.int32)
  return image, label


def load_multi_mnist(dataset_name, split_name, num_threads, buffer_size):
  k = int(dataset_name.split("-")[2])
  rgb = "rgb" in dataset_name
  unpack = functools.partial(unpack_multi_mnist_image, split_name, k, rgb)
  filename = os.path.join(FLAGS.multigan_dataset_root,
                          "%s-%s.tfrecords" % (dataset_name, split_name))

  return tf.data.TFRecordDataset(
      filename,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack, num_parallel_calls=num_threads)


def get_dataset_params():
  """Returns a dictionary containing dicts with hyper params for datasets."""

  params = {
      "clevr": {
          "input_height": 128,
          "input_width": 128,
          "output_height": 128,
          "output_width": 128,
          "c_dim": 3,
          "dataset_name": "clevr",
          "eval_test_samples": 10000
      },
  }

  # Add multi-mnist configs.
  for dataset_name in MULTI_MNIST_CONFIGS:
    c_dim = 3 if "rgb" in dataset_name else 1
    params.update({
        dataset_name: {
            "input_height": 64,
            "input_width": 64,
            "output_height": 64,
            "output_width": 64,
            "c_dim": c_dim,
            "dataset_name": dataset_name,
            "eval_test_samples": 10000
        }
    })

  return params


def get_datasets():
  """Returns a dict containing methods to load specific dataset."""

  datasets = {
      "clevr": load_clevr,
  }

  # Add multi-mnist configs.
  for dataset_name in MULTI_MNIST_CONFIGS:
    datasets[dataset_name] = load_multi_mnist

  return datasets
