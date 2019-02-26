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

"""Contains SessionRunHooks for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import logging
import tensorflow as tf


class AsyncCheckpointSaverHook(tf.contrib.tpu.AsyncCheckpointSaverHook):
  """Saves checkpoints every N steps in a asynchronous thread.

  This is the same as tf.contrib.tpu.AsyncCheckpointSaverHook but guarantees
  that there will be a checkpoint every `save_steps` steps. This helps to have
  eval results at fixed step counts, even when training is paused between
  regular checkpoint intervals.
  """

  def after_create_session(self, session, coord):
    super(AsyncCheckpointSaverHook, self).after_create_session(session, coord)
    # Interruptions to the training job can cause non-regular checkpoints
    # (between every_steps). Modify last triggered step to point to the last
    # regular checkpoint step to make sure we trigger on the next regular
    # checkpoint step.
    step = session.run(self._global_step_tensor)
    every_steps = self._timer._every_steps  # pylint: disable=protected-access
    last_triggered_step = step - step % every_steps
    self._timer.update_last_triggered_step(last_triggered_step)


class EveryNSteps(tf.train.SessionRunHook):
  """"Base class for hooks that execute callbacks every N steps.

  class MyHook(EveryNSteps):
    def __init__(self, every_n_steps):
      super(MyHook, self).__init__(every_n_steps)

    def every_n_steps_after_run(self, step, run_context, run_values):
      # Your Implementation

  If you do overwrite begin(), end(), before_run() or after_run() make sure to
  call super() at the beginning.
  """

  def __init__(self, every_n_steps):
    """Initializes an `EveryNSteps` hook.

    Args:
      every_n_steps: `int`, the number of steps to allow between callbacks.
    """
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_steps)
    self._global_step_tensor = None

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step must be created to use EveryNSteps.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    """Overrides `SessionRunHook.before_run`.

    Args:
      run_context: A `SessionRunContext` object.

    Returns:
      None or a `SessionRunArgs` object.
    """
    return tf.train.SessionRunArgs({"global_step": self._global_step_tensor})

  def after_run(self, run_context, run_values):
    """Overrides `SessionRunHook.after_run`.

    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    step = run_values.results["global_step"]
    if self._timer.should_trigger_for_step(step):
      self.every_n_steps_after_run(step, run_context, run_values)
      self._timer.update_last_triggered_step(step)

  def end(self, sess):
    step = sess.run(self._global_step_tensor)
    self.every_n_steps_after_run(step, None, None)

  def every_n_steps_after_run(self, step, run_context, run_values):
    """Callback after every n"th call to run().

    Args:
      step: Current global_step value.
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    raise NotImplementedError("Subclasses of EveryNSteps should implement "
                              "every_n_steps_after_run().")


class ReportProgressHook(EveryNSteps):
  """SessionRunHook that reports progress to a `TaskManager` instance."""

  def __init__(self, task_manager, max_steps, every_n_steps=100):
    """Create a new instance of ReportProgressHook.

    Args:
      task_manager: A `TaskManager` instance that implements report_progress().
      max_steps: Maximum number of training steps.
      every_n_steps: How frequently the hook should report progress.
    """
    super(ReportProgressHook, self).__init__(every_n_steps=every_n_steps)
    logging.info("Creating ReportProgressHook to report progress every %d "
                 "steps.", every_n_steps)
    self.max_steps = max_steps
    self.task_manager = task_manager
    self.start_time = None
    self.start_step = None

  def every_n_steps_after_run(self, step, run_context, run_values):
    if self.start_time is None:
      # First call.
      self.start_time = time.time()
      self.start_step = step
      return

    time_elapsed = time.time() - self.start_time
    steps_per_sec = float(step - self.start_step) / time_elapsed
    eta_seconds = (self.max_steps - step) / (steps_per_sec + 0.0000001)
    message = "{:.1f}% @{:d}, {:.1f} steps/s, ETA: {:.0f} min".format(
        100 * step / self.max_steps, step, steps_per_sec, eta_seconds / 60)
    logging.info("Reporting progress: %s", message)
    self.task_manager.report_progress(message)
