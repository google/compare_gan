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

"""Provides ModularGAN for GAN models with penalty loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import logging
from compare_gan import utils
from compare_gan.gans.modular_gan import ModularGAN
from compare_gan.tpu import tpu_summaries
import gin
import tensorflow as tf


@gin.configurable(blacklist=["dataset", "parameters", "model_dir"])
class GPUModularGAN(ModularGAN):
  """Base class for GANs models that support the Estimator API on GPU.

  ModularGAN unrolls the generator iterations as this increases efficiency on
  TPU. However, on GPU this can lead to OOM errors when the model and/or batch
  size is large, or when multiple discriminator iterations per generator
  iteration are performed. GPUModularGAN omits unrolling and therefore
  might prevent OOM errors on GPU.

  IMPORTANT: In contrast to ModularGAN, GPUModularGAN uses the same batch
  of generated samples to update the generator as used in the most recent
  discriminator update. This leads to additional memory savinings and is what
  used to be done in compare_gan_v2 by default.
  """

  @property
  def num_sub_steps(self):
    # No unrolling, so number of sub steps is always 1
    return 1

  def model_fn(self, features, labels, params, mode):
    """Constructs the model for the given features and mode.

    Args:
      features: A dictionary with the feature tensors.
      labels: Tensor will labels. Will be None if mode is PREDICT.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      mode: `tf.estimator.ModeKeys` value (TRAIN, EVAL, PREDICT). The mode
          should be passed to the TPUEstimatorSpec and your model should be
          build this mode.

    Returns:
      A `tf.contrib.tpu.TPUEstimatorSpec`.
    """
    logging.info("model_fn(): features=%s, labels=%s,mode=%s, params=%s",
                 features, labels, mode, params)

    if mode != tf.estimator.ModeKeys.TRAIN:
      raise ValueError("Only training mode is supported.")

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    global_step = tf.train.get_or_create_global_step()
    # Variable to count discriminator steps
    global_step_disc = tf.get_variable("global_step_discriminator",
                                       dtype=tf.int32,
                                       initializer=tf.constant(0),
                                       trainable=False)

    # Create ops for first D steps here to create the variables.
    with tf.name_scope("disc_step"):
      self.create_loss(features, labels, params,
                       is_training=is_training, reuse=False)

    # Divide trainable variables into a group for D and group for G.
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    g_vars = [var for var in t_vars if "generator" in var.name]
    if len(t_vars) != len(d_vars) + len(g_vars):
      logging.error("There variables that neither part of G or D.")
    self._check_variables(t_vars, d_vars, g_vars)

    d_optimizer = self.d_optimizer(params["use_tpu"])
    g_optimizer = self.g_optimizer(params["use_tpu"])

    # In the following each sub-step (disc_iters steps on D + one step on G)
    # depends on previous sub-steps. The optimizer ops for each step
    # depends on all the update ops (from batch norm etc.). Each update op
    # will still only be executed ones.
    deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Discriminator training.
    with tf.control_dependencies(deps):
      deps.append(d_optimizer.minimize(
          self.d_loss, var_list=d_vars, global_step=global_step_disc))

    # Clean old summaries from previous calls to model_fn().
    self._tpu_summary = tpu_summaries.TpuSummaries(self._model_dir)
    self._tpu_summary.scalar("loss/d", self.d_loss)
    with tf.name_scope("fake_images"):
      z = features["z"]
      sampled_y = None
      if self.conditional:
        sampled_y = self._get_one_hot_labels(features["sampled_labels"])
      fake_images = self.generator(
          z, y=sampled_y, is_training=True, reuse=True)
    self._add_images_to_summary(fake_images, "fake_images", params)
    self._add_images_to_summary(features["images"], "real_images", params)

    # Generator training.
    with tf.name_scope("gen_step"):
      with tf.control_dependencies(deps):
        self._tpu_summary.scalar("loss/g", self.g_loss)
        deps.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
      with tf.control_dependencies(deps):
        if self._disc_iters == 1:
          train_op = g_optimizer.minimize(self.g_loss, var_list=g_vars,
                                          global_step=global_step)
        else:
          # We should only train the generator every self.disc_iter steps.
          # We can do this using `tf.cond`. Both paths must return a tensor.
          # Our true_fn will return a tensor that depends on training the
          # generator, while the tensor from false_fn depends on nothing.
          def do_train_generator():
            actual_train_op = g_optimizer.minimize(self.g_loss, var_list=g_vars,
                                                   global_step=global_step)
            with tf.control_dependencies([actual_train_op]):
              return tf.constant(0)
          def do_not_train_generator():
            return tf.constant(0)
          train_op = tf.cond(
              tf.equal(global_step_disc % self._disc_iters, 0),
              true_fn=do_train_generator,
              false_fn=do_not_train_generator,
              name="").op
        loss = self.g_loss

    if self._g_use_ema:
      with tf.name_scope("generator_ema"):
        logging.info("Creating moving averages of weights: %s", g_vars)
        def do_update_ema():
          # The decay value is set to 0 if we're before the moving-average start
          # point, so that the EMA vars will be the normal vars.
          decay = self._ema_decay * tf.cast(
              tf.greater_equal(global_step, self._ema_start_step), tf.float32)
          ema = tf.train.ExponentialMovingAverage(decay=decay)
          return ema.apply(g_vars)
        def do_not_update_ema():
          return tf.constant(0).op

        with tf.control_dependencies([train_op]):
          train_op = tf.cond(
              tf.equal(global_step_disc % self._disc_iters, 0),
              true_fn=do_update_ema,
              false_fn=do_not_update_ema,
              name="")

    d_param_overview = utils.get_parameter_overview(d_vars, limit=None)
    g_param_overview = utils.get_parameter_overview(g_vars, limit=None)
    logging.info("Discriminator variables:\n%s", d_param_overview)
    logging.info("Generator variables:\n%s", g_param_overview)

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        host_call=self._tpu_summary.get_host_call(),
        # Estimator requires a loss which gets displayed on TensorBoard.
        # The given Tensor is evaluated but not used to create gradients.
        loss=loss,
        train_op=train_op)
