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

import collections
import contextlib
import os

from compare_gan.src import datasets
from compare_gan.src import params
from compare_gan.src.gans import ops
from compare_gan.src.gans.BEGAN import BEGAN
from compare_gan.src.gans.DRAGAN import DRAGAN
from compare_gan.src.gans.GAN import GAN
from compare_gan.src.gans.GAN import GAN_MINMAX
from compare_gan.src.gans.gans_with_penalty import GAN_PENALTY
from compare_gan.src.gans.gans_with_penalty import LSGAN_PENALTY
from compare_gan.src.gans.gans_with_penalty import WGAN_PENALTY
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
    "VAE": VAE,
    "SN_GAN": GAN_PENALTY,
    "GAN_PENALTY": GAN_PENALTY,
    "LSGAN_PENALTY": LSGAN_PENALTY,
    "WGAN_PENALTY": WGAN_PENALTY,
}

DATASETS = {"mnist": datasets.load_mnist,
            "fashion-mnist": datasets.load_fashion_mnist,
            "triangles": datasets.load_triangles_and_squares,
            "squares": datasets.load_triangles_and_squares,
            "cifar10": datasets.load_cifar10,
            "celeba": datasets.load_celeba,
            "fake": datasets.load_fake,
            "celebahq128": datasets.load_celebahq,
            "lsun-bedroom": datasets.load_lsun}

flags = tf.flags
FLAGS = flags.FLAGS
logging = tf.logging

flags.DEFINE_integer("data_reading_num_threads", 4,
                     "The number of threads used to read the dataset.")
flags.DEFINE_integer("data_reading_buffer_bytes", 128 * 1024,
                     "The buffer size used to read the dataset.")


# Seed for shuffling dataset.
DEFAULT_DATASET_SEED = 547


def load_dataset(dataset_name,
                 split_name="train",
                 num_threads=None,
                 buffer_size=None):
  """Reads files and returns a TFDataset object."""
  if not num_threads:
    num_threads = FLAGS.data_reading_num_threads
  if not buffer_size:
    buffer_size = FLAGS.data_reading_buffer_bytes
  if split_name not in ["test", "train", "val"]:
    raise ValueError("Invalid split name.")
  if dataset_name not in DATASETS:
    raise ValueError("Dataset %s is not available." % dataset_name)

  return DATASETS[dataset_name](dataset_name, split_name, num_threads,
                                buffer_size)


def create_gan(gan_type, dataset, dataset_content, options,
               checkpoint_dir, result_dir, gan_log_dir):
  """Instantiates a GAN with the requested options."""

  if gan_type not in MODELS:
    raise Exception("[!] Unrecognized GAN type: %s" % gan_type)

  if dataset not in DATASETS:
    raise ValueError("Dataset %s is not available." % dataset)

  # We use the same batch size and latent space dimension for all GANs.
  parameters = {
      "batch_size": 64,
      "z_dim": 64,
  }
  # Get the default parameters for the dataset.
  parameters.update(params.GetDatasetParameters(dataset))
  # Get the parameters provided in the argument.
  parameters.update(options)

  assert parameters["training_steps"] >= 1, (
      "Number of steps has to be positive.")
  assert parameters["save_checkpoint_steps"] >= 1, (
      "The number of steps per eval should be positive")
  assert parameters["batch_size"] >= 1, "Batch size has to be positive."
  assert parameters["z_dim"] >= 1, ("Number of latent dimensions has to be"
                                         " positive.")

  # Runtime settings for GANs.
  runtime_info = collections.namedtuple(
          'RuntimeInfo', ['checkpoint_dir', 'result_dir', 'log_dir'])

  runtime_info.checkpoint_dir = checkpoint_dir
  runtime_info.result_dir = result_dir
  runtime_info.log_dir = gan_log_dir

  return MODELS[gan_type](
      dataset_content=dataset_content,
      parameters=parameters,
      runtime_info=runtime_info)


@contextlib.contextmanager
def profile_context(tfprofile_dir):
  if "enable_tf_profile" in FLAGS and FLAGS.enable_tf_profile:
    with tf.contrib.tfprof.ProfileContext(
        tfprofile_dir, trace_steps=range(100, 200, 1), dump_steps=[200]):
      yield
  else:
    yield


def run_with_options(options, task_workdir, progress_reporter=None):
  """Runs the task with arbitrary options."""
  checkpoint_dir = os.path.join(task_workdir, "checkpoint")
  tfprofile_dir = os.path.join(task_workdir, "tfprofile")
  result_dir = os.path.join(task_workdir, "result")
  gan_log_dir = os.path.join(task_workdir, "logs")

  gan_type = options["gan_type"]
  dataset = options["dataset"]

  logging.info("Running tasks with gan_type: %s dataset: %s with parameters %s",
               gan_type, dataset, str(options))
  logging.info("Checkpoint dir: %s result_dir: %s gan_log_dir: %s",
               checkpoint_dir, result_dir, gan_log_dir)

  ops.check_folder(checkpoint_dir)
  ops.check_folder(tfprofile_dir)
  ops.check_folder(result_dir)
  ops.check_folder(gan_log_dir)

  # Set the dataset shuffling seed if specified in options.
  dataset_seed = DEFAULT_DATASET_SEED
  if "dataset_seed" in options:
    logging.info("Seeting dataset seed to %d", options["dataset_seed"])
    dataset_seed = options["dataset_seed"]

  dataset_content = load_dataset(dataset, split_name="train")
  dataset_content = dataset_content.repeat().shuffle(
      10000, seed=dataset_seed)

  with tf.Graph().as_default():
    if "tf_seed" in options:
      seed = options["tf_seed"]
      logging.info("Setting tf and np random seed to %d", seed)
      tf.set_random_seed(seed)
      np.random.seed(seed)

    with profile_context(tfprofile_dir):
      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = create_gan(
            gan_type=gan_type,
            dataset=dataset,
            dataset_content=dataset_content,
            options=options,
            checkpoint_dir=checkpoint_dir,
            result_dir=result_dir,
            gan_log_dir=gan_log_dir)
        gan.build_model()
        print " [*] Training started!"
        gan.train(sess, progress_reporter)
        print " [*] Training finished!"
