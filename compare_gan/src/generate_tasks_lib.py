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

"""Generate tasks for comparing GANs."""

from __future__ import absolute_import
from __future__ import division

import collections
import copy
import csv
import math
import os
import random

from compare_gan.src import params
from compare_gan.src import simple_task_pb2
from compare_gan.src import task_utils
import tensorflow as tf
from google.protobuf import text_format

flags = tf.flags
FLAGS = flags.FLAGS
logging = tf.logging

BATCH_SIZE = 64

ALL_GANS = [
    "GAN", "GAN_MINMAX", "WGAN", "WGAN_GP",
    "DRAGAN", "VAE", "LSGAN", "BEGAN"]

flags.DEFINE_string(
    "phase1_dir", "/tmp/phase1/",
    "This flag has to be set only if you are running phase3 experiments. "
    "It is a path to the directory with phase1 results, that will be used by "
    "phase3 to automatically extract the best hyperparameters for given model.")


def TestExp():
  """Run for one epoch over all tested GANs."""
  config = {
      "gan_type": ["GAN", "GAN_MINMAX", "WGAN", "WGAN_GP",
                   "DRAGAN", "VAE", "LSGAN", "BEGAN"],
      "dataset": ["mnist"],
      "training_steps": [60000 // BATCH_SIZE],
      "save_checkpoint_steps": [10000],
      "batch_size": [BATCH_SIZE],
      "tf_seed": [42],
      "z_dim": [64],
  }
  tasks = task_utils.CrossProduct(config)
  # Add GAN and dataset specific hyperparams.
  for task in tasks:
    task.update(GetDefaultParams(
        params.GetParameters(task["gan_type"], task["dataset"])))
  return tasks


def RepeatExp(gan_type, num_repeat):
  """Experiment where 1 fixed gan is trained with num_repeat seeds."""
  config = {
      "gan_type": [gan_type],
      "dataset": ["mnist", "fashion-mnist"],
      "training_steps": [20 * 60000 // BATCH_SIZE],
      "save_checkpoint_steps": [5 * 60000 // BATCH_SIZE],
      "batch_size": [BATCH_SIZE],
      "z_dim": [64],
      "tf_seed": range(num_repeat)
  }
  tasks = task_utils.CrossProduct(config)
  # Add GAN and dataset specific hyperparams.
  for task in tasks:
    task.update(GetDefaultParams(
        params.GetParameters(task["gan_type"], task["dataset"])))
  return tasks


def GetDefaultParams(gan_params):
  """Return the default params for a GAN (=the ones used in the paper)."""
  ret = {}
  for param_name, param_info in gan_params.iteritems():
    ret[param_name] = param_info.default
  return ret


def GetSample(gan_params):
  """Get one sample for each range specified in gan_params."""
  ret = {}
  for param_name, param_info in sorted(gan_params.items()):
    if param_info.is_discrete:
      v = random.choice(param_info.range)
    else:
      assert isinstance(param_info.default, float)
      assert param_info.range[0] <= param_info.range[1]
      v = random.uniform(param_info.range[0], param_info.range[1])
      if param_info.is_log_scale:
        v = math.pow(10, v)
    ret[param_name] = v

  return ret


def TuneParams(gan_types, dataset_name, num_samples,
               num_repeat, range_type="wide"):
  """Create a set of tasks for optimizing model score."""
  all_tasks = []
  for gan_type in gan_types:
    gan_params = params.GetParameters(gan_type, dataset_name, range_type)
    # Always include default params.
    samples = [GetDefaultParams(gan_params)]
    for _ in range(num_samples - 1):
      samples.append(GetSample(gan_params))

    for idx, sample in enumerate(samples):
      for i in range(num_repeat):
        s = copy.deepcopy(sample)
        s["gan_type"] = gan_type
        s["sample_id"] = idx  # For joins in dremel.
        s["dataset"] = dataset_name
        s["tf_seed"] = i
        if dataset_name == "cifar10":
          # 100 epochs for CIFAR100
          epoch = 50000
          s["training_steps"] = 100 * epoch // BATCH_SIZE
        elif dataset_name == "celeba":
          # 40 epochs for CelebA
          epoch = 162000
          s["training_steps"] = 40 * epoch // BATCH_SIZE
        else:
          # 20 epochs for MNIST/FASHION-MNIST
          epoch = 50000
          s["training_steps"] = 20 * epoch // BATCH_SIZE
        s["save_checkpoint_steps"] = 5 * epoch // BATCH_SIZE
        all_tasks.append(s)

  all_tasks = [collections.OrderedDict(sorted(x.items())) for x in all_tasks]
  return all_tasks


def BestFIDPerModel(csv_file_pattern):
  """Returns a dict with best score and checkpoint per model."""
  best_score_per_gan = {}
  for csv_file_path in tf.gfile.Glob(csv_file_pattern):
    with tf.gfile.Open(csv_file_path, "r") as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        model = row["model"]
        fid_score = float(row["fid_score"])
        if best_score_per_gan.get(model, (100000.0, ""))[0] > fid_score:
          best_score_per_gan[model] = (fid_score, row["checkpoint_path"])
  return best_score_per_gan


def GetTaskProtoFromCheckpoint(checkpoint_path):
  task_file = os.path.join(os.path.dirname(checkpoint_path), "../task")
  if not tf.gfile.Exists(task_file):
    logging.warning("Cant find task_file at %s", task_file)
    assert False, "Task file %s not found." % task_file
  content = tf.gfile.Open(task_file, mode="r").read()
  return text_format.Parse(content, simple_task_pb2.Task())


def Phase3Auto(model_list, csv_file_pattern, num_repeat):
  """Phase 3: repeat best model N times with different seeds."""
  all_tasks = []
  best_fid_per_model = BestFIDPerModel(csv_file_pattern)
  all_checkpoints = []
  for model in model_list:
    all_checkpoints.append(best_fid_per_model[model][1])
    logging.info("Model: %s, Best task: %s", model,
                 best_fid_per_model[model][1])

  for checkpoint in all_checkpoints:
    task_proto = GetTaskProtoFromCheckpoint(checkpoint)
    options = task_utils.ParseOptions(task_proto)
    # Map unicode to string.
    for k in options.keys():
      if isinstance(options[k], unicode):
        options[k] = str(options[k])

    for seed in range(num_repeat):
      s = copy.deepcopy(options)
      s["tf_seed"] = seed
      all_tasks.append(s)

  all_tasks = [collections.OrderedDict(sorted(x.items())) for x in all_tasks]
  return all_tasks


def AdamVsRmsprop(datasets):
  """WGAN uses RMSProp, how much does it affect the scores?."""
  ret = []
  for dataset in datasets:
    # Make sure we have the same parameters as in phase1.
    random.seed(123)
    tasks = TuneParams(
        ALL_GANS, dataset, num_samples=100, num_repeat=1, range_type="wide")
    for optimizer in ["adam", "rmsprop"]:
      for i in range(len(tasks)):
        if tasks[i]["gan_type"] != "WGAN":
          continue
        task = copy.deepcopy(tasks[i])
        task["optimizer"] = optimizer
        ret.append(task)

  return ret


def GetTasks(experiment):
  """Get a list of tasks to run and eval for the given experiment name."""
  random.seed(123)

  if experiment == "test":
    return TestExp()
  # PHASE 1 EXPERIMENTS BELOW (WIDE RANGE)
  elif experiment == "phase1_gan8_mnist_sample100_rep1":
    return TuneParams(ALL_GANS, "mnist", num_samples=100, num_repeat=1,
                      range_type="wide")
  elif experiment == "phase1_gan8_fashionmnist_sample100_rep1":
    return TuneParams(ALL_GANS, "fashion-mnist", num_samples=100, num_repeat=1,
                      range_type="wide")
  elif experiment == "phase1_gan8_cifar_sample100_rep1":
    return TuneParams(ALL_GANS, "cifar10", num_samples=100, num_repeat=1,
                      range_type="wide")
  elif experiment == "phase1_gan8_celeba_sample100_rep1":
    return TuneParams(ALL_GANS, "celeba", num_samples=100, num_repeat=1,
                      range_type="wide")
  elif experiment == "phase1_gan8_triangles_sample100_rep1":
    return TuneParams(ALL_GANS, "triangles", num_samples=100, num_repeat=1,
                      range_type="wide")
  # PHASE 2 EXPERIMENTS BELOW (NARROW RANGE)
  elif experiment == "phase2_gan8_mnist_sample50_rep1":
    return TuneParams(ALL_GANS, "mnist", num_samples=50, num_repeat=1,
                      range_type="narrow")
  elif experiment == "phase2_gan8_fashionmnist_sample50_rep1":
    return TuneParams(ALL_GANS, "fashion-mnist", num_samples=50, num_repeat=1,
                      range_type="narrow")
  elif experiment == "phase2_gan8_cifar_sample50_rep1":
    return TuneParams(ALL_GANS, "cifar10", num_samples=50, num_repeat=1,
                      range_type="narrow")
  elif experiment == "phase2_gan8_celeba_sample50_rep1":
    return TuneParams(ALL_GANS, "celeba", num_samples=50, num_repeat=1,
                      range_type="narrow")
  # PHASE 3 EXPERIMENTS BELOW (RETRY BEST FROM WIDE RANGE)
  elif experiment == "phase3_mnist":
    return Phase3Auto(
        ALL_GANS,
        os.path.join(FLAGS.phase1_dir,
                     "phase1_gan8_mnist_sample100_rep1/task_num_*/scores.csv"),
        num_repeat=50)
  elif experiment == "phase3_fashionmnist":
    return Phase3Auto(
        ALL_GANS,
        os.path.join(
            FLAGS.phase1_dir,
            "phase1_gan8_fashionmnist_sample100_rep1/task_num_*/scores.csv"),
        num_repeat=50)
  elif experiment == "phase3_cifar":
    return Phase3Auto(
        ALL_GANS,
        os.path.join(FLAGS.phase1_dir,
                     "phase1_gan8_cifar_sample100_rep1/task_num_*/scores.csv"),
        num_repeat=50)
  elif experiment == "phase3_celeba":
    return Phase3Auto(
        ALL_GANS,
        os.path.join(FLAGS.phase1_dir,
                     "phase1_gan8_celeba_sample100_rep1/task_num_*/scores.csv"),
        num_repeat=50)
  # Additional experiments.
  elif experiment == "adam_vs_rmsprop_cifar_celeba":
    return AdamVsRmsprop(datasets=["cifar10", "celeba"])
  else:
    raise ValueError("Unknown experiment %s" % experiment)

