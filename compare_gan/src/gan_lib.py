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

"""Library used for training various flavors of GANs on various datasets."""

import os
from compare_gan.src import params

from compare_gan.src.gans import ops
from compare_gan.src.gans.BEGAN import BEGAN
from compare_gan.src.gans.DRAGAN import DRAGAN
from compare_gan.src.gans.GAN import GAN
from compare_gan.src.gans.GAN import GAN_MINMAX
from compare_gan.src.gans.LSGAN import LSGAN
from compare_gan.src.gans.VAE import VAE
from compare_gan.src.gans.WGAN import WGAN
from compare_gan.src.gans.WGAN_GP import WGAN_GP

import numpy as np
import tensorflow as tf

MODELS = {
    "GAN": GAN,
    "GAN_MINMAX": GAN_MINMAX,
    "WGAN": WGAN,
    "WGAN_GP": WGAN_GP,
    "DRAGAN": DRAGAN,
    "LSGAN": LSGAN,
    "BEGAN": BEGAN,
    "VAE": VAE
}

DATASETS = ["mnist", "fashion-mnist", "triangles", "squares", "cifar10",
            "celeba", "celeba20k"]

flags = tf.flags
FLAGS = flags.FLAGS
logging = tf.logging

flags.DEFINE_string("dataset_root",
                    "/tmp/"
                    "datasets/gan_compare/",
                    "Folder which contains all datasets.")

# Seed for shuffling dataset.
DEFAULT_DATASET_SEED = 547


def load_convex(dataset_name):
  folder_path = os.path.join(FLAGS.dataset_root, "convex")
  file_path = "%s/%s.npz" % (folder_path, dataset_name)
  with tf.gfile.Open(file_path, "r") as infile:
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


def get_sharded_filenames(prefix, num_shards):
  return [
      os.path.join(FLAGS.dataset_root,
                   prefix + "-%05d-of-%05d" % (i, num_shards))
      for i in xrange(0, num_shards)
  ]


def load_dataset(dataset_name, split_name="train"):
  """Reads files and returns Dataset object with pair: image, label."""
  if dataset_name in ["triangles", "squares"]:
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

  if split_name not in ["test", "train"]:
    raise ValueError("Invalid split name.")

  if dataset_name == "mnist":
    if split_name == "train":
      filenames = get_sharded_filenames("image_mnist-train", 10)
    else:
      filenames = get_sharded_filenames("image_mnist-dev", 1)
    # Image dim: 28,28,1 range: 0..1 label: int32
    return tf.data.TFRecordDataset(filenames).map(unpack_png_image)
  elif dataset_name == "fashion-mnist":
    if split_name == "train":
      filenames = get_sharded_filenames("image_fashion_mnist-train", 10)
    else:
      filenames = get_sharded_filenames("image_fashion_mnist-dev", 1)
    # Image dim: 28,28,1 range: 0..1 label: int32
    return tf.data.TFRecordDataset(filenames).map(unpack_png_image)
  elif dataset_name == "cifar10":
    if split_name == "train":
      filenames = get_sharded_filenames("image_cifar10-train", 10)
    else:
      filenames = get_sharded_filenames("image_cifar10-dev", 1)
    # Image dim: 32,32,3 range: 0..1 label: int32
    return tf.data.TFRecordDataset(filenames).map(unpack_png_image)
  elif dataset_name == "celeba":
    if split_name == "train":
      filenames = get_sharded_filenames("image_celeba_tune-train", 100)
    else:
      filenames = get_sharded_filenames("image_celeba_tune-dev", 10)
    # Image dim: 64,64,3 range: 0..1 label: 0 (fixed)
    return tf.data.TFRecordDataset(filenames).map(unpack_celeba_image)

  raise NotImplementedError("Unknown dataset")


def create_gan(gan_type, dataset, sess, dataset_content, options,
               checkpoint_dir, result_dir, gan_log_dir):
  """Instantiates a GAN with the requested options."""

  if gan_type not in MODELS:
    raise Exception("[!] Unrecognized GAN type: %s" % gan_type)
  if dataset not in DATASETS:
    raise Exception("[!] Unrecognized dataset: %s" % dataset)

  # We use the same batch size and latent space dimension for all GANs.
  training_params = {
      "batch_size": 64,
      "z_dim": 64,
  }
  training_params.update(options)
  dataset_params = params.GetDatasetParameters(dataset)
  dataset_params.update(options)

  assert training_params["training_steps"] >= 1, (
      "Number of steps has to be positive.")
  assert training_params["save_checkpoint_steps"] >= 1, (
      "The number of steps per eval should be positive")
  assert training_params["batch_size"] >= 1, "Batch size has to be positive."
  assert training_params["z_dim"] >= 1, ("Number of latent dimensions has to be"
                                         " positive.")

  return MODELS[gan_type](
      sess=sess,
      dataset_content=dataset_content,
      dataset_parameters=dataset_params,
      training_parameters=training_params,
      checkpoint_dir=checkpoint_dir,
      result_dir=result_dir,
      log_dir=gan_log_dir)


def run_with_options(options, task_workdir):
  """Runs the task with arbitrary options."""

  # Set the dataset shuffling seed if specified in options.
  dataset_seed = DEFAULT_DATASET_SEED
  if "dataset_seed" in options:
    logging.info("Seeting dataset seed to %d", options["dataset_seed"])
    dataset_seed = options["dataset_seed"]

  if "tf_seed" in options:
    seed = options["tf_seed"]
    logging.info("Setting tf random seed to %d", seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

  checkpoint_dir = os.path.join(task_workdir, "checkpoint")
  result_dir = os.path.join(task_workdir, "result")
  gan_log_dir = os.path.join(task_workdir, "logs")

  gan_type = options["gan_type"]
  dataset = options["dataset"]

  logging.info("Running tasks with gan_type: %s dataset: %s with parameters %s",
               gan_type, dataset, str(params))
  logging.info("Checkpoint dir: %s result_dir: %s gan_log_dir: %s",
               checkpoint_dir, result_dir, gan_log_dir)

  ops.check_folder(checkpoint_dir)
  ops.check_folder(result_dir)
  ops.check_folder(gan_log_dir)

  dataset_content = load_dataset(dataset, split_name="train")

  dataset_content = dataset_content.repeat().shuffle(
      10000, seed=dataset_seed)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    gan = create_gan(
        gan_type=gan_type,
        dataset=dataset,
        sess=sess,
        dataset_content=dataset_content,
        options=options,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        gan_log_dir=gan_log_dir)
    gan.build_model()
    print " [*] Training started!"
    gan.train()
    print " [*] Training finished!"
