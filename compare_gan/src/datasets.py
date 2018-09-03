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

"""Dataset loading code for compare_gan."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from six.moves import range
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dataset_root",
    "/tmp/datasets/gan_compare/",
    "Folder which contains all datasets.")


def load_convex(dataset_name):
  folder_path = os.path.join(FLAGS.dataset_root, "convex")
  file_path = "%s/%s.npz" % (folder_path, dataset_name)
  with tf.gfile.Open(file_path, "rb") as infile:
    data = np.load(infile)
    features, labels = data["x"], data["y"]
    return features, labels


def unpack_png_image(image_data):
  """Returns an image and a label. 0-1 range."""
  value = tf.parse_single_example(
      image_data,
      features={"image/encoded": tf.FixedLenFeature([], tf.string),
                "image/class/label": tf.FixedLenFeature([], tf.int64)})
  image = tf.image.decode_png(value["image/encoded"])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.cast(value["image/class/label"], tf.int32)
  return image, label


def unpack_raw_image(image_data):
  """Returns an image and a label. 0-1 range."""
  value = tf.parse_single_example(
      image_data,
      features={"image": tf.FixedLenFeature([], tf.string)})
  image = tf.decode_raw(value["image"], out_type=tf.uint8)
  image = tf.reshape(image, [128, 128, 3])
  image = tf.cast(image, tf.float32) / 255.0
  return image, tf.constant(0, dtype=tf.int32)


def unpack_celeba_image(image_data):
  """Returns 64x64x3 image and constant label."""
  value = tf.parse_single_example(
      image_data,
      features={"image/encoded": tf.FixedLenFeature([], tf.string)})
  image = tf.image.decode_png(value["image/encoded"])
  image = tf.image.resize_image_with_crop_or_pad(image, 160, 160)
  # Note: possibly consider using NumPy's imresize(image, (64, 64))
  image = tf.image.resize_images(image, [64, 64])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.constant(0, dtype=tf.int32)
  return image, label


def get_sharded_filenames(prefix, num_shards, range_start=0, range_end=None):
  """Retrieves sharded file names."""
  if range_end is None:
    range_end = num_shards
  return [
      os.path.join(FLAGS.dataset_root,
                   "%s-%05d-of-%05d" % (prefix, i, num_shards))
      for i in range(range_start, range_end)
  ]


def load_triangles_and_squares(dataset_name, split_name, num_threads,
                               buffer_size):
  """Loads the triangle and squares dataset."""
  del num_threads, buffer_size
  features, labels = load_convex(dataset_name)
  assert features.shape[0] == 80000
  assert labels.shape[0] == features.shape[0]
  if split_name == "train":
    return tf.data.Dataset.from_tensor_slices(
        (features[:60000], labels[:60000]))
  elif split_name == "test":
    return tf.data.Dataset.from_tensor_slices(
        (features[-20000:-10000], labels[-20000:-10000]))
  else:
    return tf.data.Dataset.from_tensor_slices(
        (features[-10000:], labels[-10000:]))


def load_fake(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name, split_name
  del num_threads, buffer_size
  # Fake dataset for unittests.
  return tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(np.random.uniform(
          size=(100, 64, 64, 1))),
      tf.data.Dataset.from_tensor_slices(np.zeros(shape=(100)))))


def load_mnist(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  if split_name == "train":
    filenames = get_sharded_filenames("image_mnist-train", 10)
  else:
    filenames = get_sharded_filenames("image_mnist-dev", 1)
  # Image dim: 28,28,1 range: 0..1 label: int32
  return tf.data.TFRecordDataset(
      filenames,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack_png_image, num_parallel_calls=num_threads)


def load_fashion_mnist(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  if split_name == "train":
    filenames = get_sharded_filenames("image_fashion_mnist-train", 10)
  else:
    filenames = get_sharded_filenames("image_fashion_mnist-dev", 1)
  # Image dim: 28,28,1 range: 0..1 label: int32
  return tf.data.TFRecordDataset(
      filenames,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack_png_image, num_parallel_calls=num_threads)


def load_cifar10(dataset_name, split_name, num_threads, buffer_size):
  del dataset_name
  if split_name == "train":
    filenames = get_sharded_filenames("image_cifar10-train", 10)
  else:
    filenames = get_sharded_filenames("image_cifar10-dev", 1)
  # Image dim: 32,32,3 range: 0..1 label: int32
  return tf.data.TFRecordDataset(
      filenames,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack_png_image, num_parallel_calls=num_threads)


def load_celeba(dataset_name, split_name, num_threads, buffer_size):
  """Returns Celeba-HQ as a TFRecordDataset."""
  del dataset_name
  if split_name == "train":
    filenames = get_sharded_filenames("image_celeba-train", 100)
  else:
    filenames = get_sharded_filenames("image_celeba-dev", 10)
  # Image dim: 64,64,3 range: 0..1 label: 0 (fixed)
  return tf.data.TFRecordDataset(
      filenames,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack_celeba_image, num_parallel_calls=num_threads)


def unpack_lsun_image(image_data):
  """Returns a LSUN-Bedrooms image as a 128x128x3 Tensor, with label 0."""
  value = tf.parse_single_example(
      image_data,
      features={"image/encoded": tf.FixedLenFeature([], tf.string)})
  image = tf.image.decode_jpeg(value["image/encoded"], channels=3)
  image = tf.image.resize_image_with_crop_or_pad(
      image, target_height=128, target_width=128)
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.constant(0, dtype=tf.int32)
  return image, label


def load_lsun(dataset_name, split_name, num_threads, buffer_size):
  """Returns LSUN Bedrooms as a TFRecordDataset."""
  del dataset_name
  # The eval set is too small (only 300 examples) so we're using the last
  # shard from the training set as our eval (approx. 30k examples).
  if split_name == "train":
    range_start = 0
    range_end = 99
  else:
    range_start = 99
    range_end = 100
  filenames = get_sharded_filenames(
      "image_lsun_bedrooms-train", num_shards=100,
      range_start=range_start, range_end=range_end)
  # Image dim: 128, 128 ,3 range: 0..1 label: 0 (fixed)
  return tf.data.TFRecordDataset(
      filenames,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack_lsun_image, num_parallel_calls=num_threads)


def unpack_celebahq_image(record):
  """Returns a Celeba-HQ image as a Tensor, with label 0."""
  record = tf.parse_single_example(
      record,
      features={
          "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
      })
  image = tf.image.decode_png(record["image/encoded"])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.constant(0, dtype=tf.int32)
  return image, label


CELEBA_AVAILABLE_RESOLUTIONS = ["128", "256"]


def load_celebahq(dataset_name, split_name, num_threads, buffer_size):
  """Returns Celeba-HQ as a TFRecordDataset."""
  resolution = dataset_name[-3:]

  if resolution not in CELEBA_AVAILABLE_RESOLUTIONS:
    raise ValueError("Resolution not available for CelebaHQ: %s" % dataset_name)

  # No default split for train/valid for CelebA-HQ.
  if split_name == "train":
    range_start = 0
    range_end = 90
  else:
    range_start = 90
    range_end = 100
  filenames = get_sharded_filenames("image_celebahq-%s" % resolution,
                                    num_shards=100, range_start=range_start,
                                    range_end=range_end)
  # Image dim: 128, 128, 3 range: 0..1 label: 0 (fixed)
  # Image dim: 256, 256, 3 range: 0..1 label: 0 (fixed)
  return tf.data.TFRecordDataset(
      filenames,
      buffer_size=buffer_size,
      num_parallel_reads=num_threads).map(
          unpack_celebahq_image, num_parallel_calls=num_threads)
