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

"""Binary to train and evaluate a single GAN configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import re
import time

from absl import flags
from absl import logging
from compare_gan import datasets
from compare_gan import eval_gan_lib
from compare_gan import hooks
from compare_gan.gans import utils
from compare_gan.metrics import fid_score as fid_score_lib
from compare_gan.metrics import inception_score as inception_score_lib
import gin.tf
import numpy as np
import six
import tensorflow as tf


FLAGS = flags.FLAGS


class _DummyParserDelegate(gin.config_parser.ParserDelegate):
  """Dummy class required to parse Gin configs.

  Our use case (just get the config as dictionary) does not require real
  implementations the two methods.
  """

  def configurable_reference(self, scoped_name, evaluate):
    return scoped_name

  def macro(self, scoped_name):
    return scoped_name


def _parse_gin_config(config_path):
  """Parses a Gin config into a dictionary. All values are strings."""
  with tf.gfile.Open(config_path) as f:
    config_str = f.read()
  parser = gin.config_parser.ConfigParser(config_str, _DummyParserDelegate())
  config = {}
  for statement in parser:
    if not isinstance(statement, gin.config_parser.ImportStatement):
      name = statement.scope + "/" if statement.scope else ""
      name = statement.selector + "." + statement.arg_name
      config[name] = statement.value
  return config


@gin.configurable("options")
def get_options_dict(batch_size=gin.REQUIRED,
                     gan_class=gin.REQUIRED,
                     architecture=gin.REQUIRED,
                     training_steps=gin.REQUIRED,
                     discriminator_normalization=None,
                     lamba=1,
                     disc_iters=1,
                     z_dim=128):
  """Parse legacy options from Gin configurations into a Python dict.

  Args:
    batch_size: The (global) batch size to use. On TPUs each core will get a
      fraction of this.
    gan_class: References to the GAN classe to use. This must implement the
      AbstractGAN interface.
    architecture: Name of the architecuter to use for G and D. This should be
      value from consts.ARCHITECTURES and be supported by the GAN class.
    training_steps: The number of training steps. These are discriminator steps.
    discriminator_normalization: Deprecated. Ignored, but kept to read old
      configs.
    lamba: Weight for gradient penalty.
    disc_iters: How often the discriminator is trained before training G for one
      step. G will be trained for `training_steps // disc_iters` steps.
    z_dim: Length of the latent variable z fed to the generator.

  Returns:
    A Python dictionary with the options.
  """
  del discriminator_normalization
  return {
      "use_tpu": FLAGS.use_tpu,  # For compatibility with AbstractGAN.
      "batch_size": batch_size,
      "gan_class": gan_class,
      "architecture": architecture,
      "training_steps": training_steps,
      "lambda": lamba,  # Different spelling intended.
      "disc_iters": disc_iters,
      "z_dim": z_dim,
  }


class TaskManager(object):
  """Interface for managing a task."""

  def __init__(self, model_dir):
    self._model_dir = model_dir

  @property
  def model_dir(self):
    return self._model_dir

  def mark_training_done(self):
    with tf.gfile.Open(os.path.join(self.model_dir, "TRAIN_DONE"), "w") as f:
      f.write("")

  def is_training_done(self):
    return tf.gfile.Exists(os.path.join(self.model_dir, "TRAIN_DONE"))

  def add_eval_result(self, checkpoint_path, result_dict, default_value):
    pass

  def get_checkpoints_with_results(self):
    return set()

  def unevaluated_checkpoints(self, timeout=0, eval_every_steps=None):
    """Generator for checkpoints without evaluation results.

    Args:
      timeout: Optional timeout for waiting for new checkpoints. Set this to
        do continious evaluation.
      eval_every_steps: Only evaluate checkpoints from steps divisible by this
                         integer.

    Yields:
      Path to checkpoints that have not yet been evaluated.
    """
    logging.info("Looking for checkpoints in %s", self._model_dir)
    evaluated_checkpoints = self.get_checkpoints_with_results()
    last_eval = time.time()
    while True:
      unevaluated_checkpoints = []
      checkpoint_state = tf.train.get_checkpoint_state(self.model_dir)
      if checkpoint_state:
        checkpoints = set(checkpoint_state.all_model_checkpoint_paths)
        # Remove already evaluated checkpoints and sort ascending by step
        # number.
        unevaluated_checkpoints = checkpoints - evaluated_checkpoints
        step_and_ckpt = sorted(
            [(int(x.split("-")[-1]), x) for x in unevaluated_checkpoints])
        if eval_every_steps:
          step_and_ckpt = [(step, ckpt) for step, ckpt in step_and_ckpt
                           if step > 0 and step % eval_every_steps == 0]
        unevaluated_checkpoints = [ckpt for _, ckpt in step_and_ckpt]
        logging.info(
            "Found checkpoints: %s\nEvaluated checkpoints: %s\n"
            "Unevaluated checkpoints: %s", checkpoints, evaluated_checkpoints,
            unevaluated_checkpoints)
      for checkpoint_path in unevaluated_checkpoints:
        yield checkpoint_path
      if unevaluated_checkpoints:
        evaluated_checkpoints |= set(unevaluated_checkpoints)
        last_eval = time.time()
        continue
      # No new checkpoints, timeout or stop if training finished. Otherwise
      # wait 1 minute.
      if time.time() - last_eval > timeout or self.is_training_done():
        break
      time.sleep(60)

  def report_progress(self, message):
    pass


class TaskManagerWithCsvResults(TaskManager):
  """Task Manager that writes results to a CSV file."""

  def __init__(self, model_dir, score_file=None):
    super(TaskManagerWithCsvResults, self).__init__(model_dir)
    if score_file is None:
      score_file = os.path.join(model_dir, "scores.csv")
    self._score_file = score_file

  def _get_config_for_step(self, step):
    """Returns the latest operative config for the global step as dictionary."""
    saved_configs = tf.gfile.Glob(
        os.path.join(self.model_dir, "operative_config-*.gin"))
    get_step = lambda fn: int(re.findall(r"operative_config-(\d+).gin", fn)[0])
    config_steps = [get_step(fn) for fn in saved_configs]
    assert config_steps
    last_config_step = sorted([s for s in config_steps if s <= step])[-1]
    config_path = os.path.join(
        self.model_dir, "operative_config-{}.gin".format(last_config_step))
    return _parse_gin_config(config_path)

  def add_eval_result(self, checkpoint_path, result_dict, default_value):
    step = os.path.basename(checkpoint_path).split("-")[-1]
    config = self._get_config_for_step(step)
    csv_header = (
        ["checkpoint_path", "step"] + sorted(result_dict) + sorted(config))
    write_header = not tf.gfile.Exists(self._score_file)
    if write_header:
      with tf.gfile.Open(self._score_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction="ignore")
        writer.writeheader()
    row = dict(checkpoint_path=checkpoint_path, step=step, **config)
    for k, v in six.iteritems(result_dict):
      if isinstance(v, float):
        v = "{:.3f}".format(v)
      row[k] = v
    with tf.gfile.Open(self._score_file, "a") as f:
      writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction="ignore")
      writer.writerow(row)

  def get_checkpoints_with_results(self):
    if not tf.gfile.Exists(self._score_file):
      return set()
    with tf.gfile.Open(self._score_file) as f:
      reader = csv.DictReader(f)
      return {r["checkpoint_path"] for r in reader}
    return set()


def _run_eval(module_spec, checkpoints, task_manager, run_config,
              use_tpu, num_averaging_runs):
  """Evaluates the given checkpoints and add results to a result writer.

  Args:
    module_spec: `ModuleSpec` of the model.
    checkpoints: Generator for for checkpoint paths.
    task_manager: `TaskManager`. init_eval() will be called before adding
      results.
    run_config: `RunConfig` to use. Values for master and tpu_config are
      currently ignored.
    use_tpu: Whether to use TPU for evaluation.
    num_averaging_runs: Determines how many times each metric is computed.
  """
  # By default, we compute FID and Inception scores. Other tasks defined in
  # the metrics folder (such as the one in metrics/kid_score.py) can be added
  # to this list if desired.
  eval_tasks = [
      inception_score_lib.InceptionScoreTask(),
      fid_score_lib.FIDScoreTask()
  ]
  logging.info("eval_tasks: %s", eval_tasks)

  for checkpoint_path in checkpoints:
    step = os.path.basename(checkpoint_path).split("-")[-1]
    if step == 0:
      continue
    export_path = os.path.join(run_config.model_dir, "tfhub", str(step))
    if not tf.gfile.Exists(export_path):
      module_spec.export(export_path, checkpoint_path=checkpoint_path)
    default_value = -1.0
    try:
      result_dict = eval_gan_lib.evaluate_tfhub_module(
          export_path, eval_tasks, use_tpu=use_tpu,
          num_averaging_runs=num_averaging_runs)
    except ValueError as nan_found_error:
      result_dict = {}
      logging.exception(nan_found_error)
      default_value = eval_gan_lib.NAN_DETECTED

    logging.info("Evaluation result for checkpoint %s: %s (default value: %s)",
                 checkpoint_path, result_dict, default_value)
    task_manager.add_eval_result(checkpoint_path, result_dict, default_value)


def run_with_schedule(schedule, run_config, task_manager, options, use_tpu,
                      num_eval_averaging_runs=1, eval_every_steps=-1):
  """Run the schedule with the given options.

  Available schedules:
  - train: Train up to options["training_steps"], continuing from existing
      checkpoints if available.
  - eval_after_train: First train up to options["training_steps"] then
      evaluate all checkpoints.
  - continuous_eval: Waiting for new checkpoints and evaluate them as they
      become available. This is meant to run in parallel with a job running
      the training schedule but can also run after it.

  Args:
    schedule: Schedule to run. One of: train, continuous_eval, train_and_eval.
    run_config: `tf.contrib.tpu.RunConfig` to use.
    task_manager: `TaskManager` for this run.
    options: Python dictionary will run parameters.
    use_tpu: Boolean whether to use TPU.
    num_eval_averaging_runs: Determines how many times each metric is computed.
    eval_every_steps: Integer determining which checkpoints to evaluate.
  """
  logging.info("Running schedule '%s' with options: %s", schedule, options)
  if run_config.tf_random_seed:
    logging.info("Setting NumPy random seed to %s.", run_config.tf_random_seed)
    np.random.seed(run_config.tf_random_seed)

  result_dir = os.path.join(run_config.model_dir, "result")
  utils.check_folder(result_dir)

  dataset = datasets.get_dataset()
  gan = options["gan_class"](dataset=dataset,
                             parameters=options,
                             model_dir=run_config.model_dir)

  if schedule not in {"train", "eval_after_train", "continuous_eval"}:
    raise ValueError("Schedule {} not supported.".format(schedule))
  if schedule in {"train", "eval_after_train"}:
    train_hooks = [
        gin.tf.GinConfigSaverHook(run_config.model_dir),
        hooks.ReportProgressHook(task_manager,
                                 max_steps=options["training_steps"]),
    ]
    if run_config.save_checkpoints_steps:
      # This replaces the default checkpoint saver hook in the estimator.
      logging.info("Using AsyncCheckpointSaverHook.")
      train_hooks.append(
          hooks.AsyncCheckpointSaverHook(
              checkpoint_dir=run_config.model_dir,
              save_steps=run_config.save_checkpoints_steps))
      # (b/122782388): Remove hotfix.
      run_config = run_config.replace(save_checkpoints_steps=1000000)
    estimator = gan.as_estimator(
        run_config, batch_size=options["batch_size"], use_tpu=use_tpu)
    estimator.train(
        input_fn=gan.input_fn,
        max_steps=options["training_steps"],
        hooks=train_hooks)
    task_manager.mark_training_done()

  if schedule == "continuous_eval":
    # Continuous eval with up to 24 hours between checkpoints.
    checkpoints = task_manager.unevaluated_checkpoints(
        timeout=24 * 3600, eval_every_steps=eval_every_steps)
  if schedule == "eval_after_train":
    checkpoints = task_manager.unevaluated_checkpoints(
        eval_every_steps=eval_every_steps)
  if schedule in {"continuous_eval", "eval_after_train"}:
    _run_eval(
        gan.as_module_spec(),
        checkpoints=checkpoints,
        task_manager=task_manager,
        run_config=run_config,
        use_tpu=use_tpu,
        num_averaging_runs=num_eval_averaging_runs)
