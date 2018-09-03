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

"""Base class for all GANs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import os
import time

from compare_gan.src.gans import consts
from compare_gan.src.gans import ops
from compare_gan.src.gans.ops import lrelu, batch_norm, linear, conv2d, deconv2d

import numpy as np
from six.moves import range
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

PREFETCH_NUM_BATCHES = 32


class AbstractGAN(object):
  """Base class for all GANs."""

  def __init__(self, model_name, dataset_content, parameters, runtime_info):
    super(AbstractGAN, self).__init__()
    self.model_name = model_name
    self.parameters = parameters
    self.use_tpu = parameters.get("use_tpu", False)

    # Initialize training-specific parameters.
    self.training_steps = int(parameters["training_steps"])
    self.save_checkpoint_steps = int(
        parameters["save_checkpoint_steps"])
    self.batch_size = parameters["batch_size"]
    self.learning_rate = parameters["learning_rate"]
    self.beta1 = parameters.get("beta1", 0.5)
    self.z_dim = parameters["z_dim"]
    self.discriminator_normalization = parameters["discriminator_normalization"]
    if self.discriminator_normalization not in consts.NORMALIZERS:
      raise ValueError(
          "Normalization not recognized: %s" % self.discriminator_normalization)

    # Output folders and checkpoints.
    self.checkpoint_dir = runtime_info.checkpoint_dir
    self.result_dir = runtime_info.result_dir
    self.log_dir = runtime_info.log_dir
    self.max_checkpoints_to_keep = 1000

    # Input and output shapes.
    self.input_height = parameters["input_height"]
    self.input_width = parameters["input_width"]
    self.output_height = parameters["output_height"]
    self.output_width = parameters["output_width"]
    self.c_dim = parameters["c_dim"]
    self.dataset_name = parameters["dataset_name"]
    self.dataset_content = dataset_content

    # Number of discriminator iterations per one iteration of the generator.
    self.disc_iters = parameters.get("disc_iters", 1)
    self.unroll_disc_iters = parameters.get("unroll_disc_iters", self.use_tpu)

  @property
  def num_sub_steps(self):
    # To support disc_iters > 1 for TPU we unroll the graph to train
    # multiple sub steps in one mini-batch. Each mini-batch will do
    # self.disc_iters sub steps, each training the discriminator and increases
    # global_step by one. After the last step we train the generator once.
    if self.unroll_disc_iters:
      return self.disc_iters
    return 1

  def discriminator(self, x, is_training, reuse=False):
    """Discriminator architecture based on InfoGAN.

    Args:
      x: input images, shape [bs, h, w, channels]
      is_training: boolean, are we in train or eval model.
      reuse: boolean, should params be re-used.

    Returns:
      out: a float (in [0, 1]) with discriminator prediction
      out_logit: the value "out" before sigmoid
      net: the architecture
    """
    sn = self.discriminator_normalization == consts.SPECTRAL_NORM
    with tf.variable_scope("discriminator", reuse=reuse):
      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = conv2d(
          x, 64, 4, 4, 2, 2, name="d_conv1", use_sn=sn)  # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = conv2d(
          net, 128, 4, 4, 2, 2, name="d_conv2",
          use_sn=sn)  # [bs, h/4, w/4, 128]
      if self.discriminator_normalization == consts.BATCH_NORM:
        net = batch_norm(net, is_training=is_training, scope="d_bn2")
      net = lrelu(net)
      net = tf.reshape(net, [self.batch_size, -1])  # [bs, h * w * 8]
      net = linear(net, 1024, scope="d_fc3", use_sn=sn)  # [bs, 1024]
      if self.discriminator_normalization == consts.BATCH_NORM:
        net = batch_norm(net, is_training=is_training, scope="d_bn3")
      net = lrelu(net)
      out_logit = linear(net, 1, scope="d_fc4", use_sn=sn)  # [bs, 1]
      out = tf.nn.sigmoid(out_logit)
      return out, out_logit, net

  def generator(self, z, is_training, reuse=False):
    height = self.input_height
    width = self.input_width
    batch_size = self.batch_size
    with tf.variable_scope("generator", reuse=reuse):
      net = linear(z, 1024, scope="g_fc1")
      net = batch_norm(net, is_training=is_training, scope="g_bn1")
      net = lrelu(net)
      net = linear(net, 128 * (height // 4) * (width // 4), scope="g_fc2")
      net = batch_norm(net, is_training=is_training, scope="g_bn2")
      net = lrelu(net)
      net = tf.reshape(net, [batch_size, height // 4, width // 4, 128])
      net = deconv2d(net, [batch_size, height // 2, width // 2, 64],
                     4, 4, 2, 2, name="g_dc3")
      net = batch_norm(net, is_training=is_training, scope="g_bn3")
      net = lrelu(net)
      net = deconv2d(net, [batch_size, height, width, self.c_dim],
                     4, 4, 2, 2, name="g_dc4")
      out = tf.nn.sigmoid(net)
      return out

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                self.batch_size, self.z_dim)

  def save(self, checkpoint_dir, step, sess):
    if not tf.gfile.IsDirectory(checkpoint_dir):
      tf.gfile.MakeDirs(checkpoint_dir)

    self.saver.save(
        sess,
        os.path.join(checkpoint_dir, self.model_name + ".model"),
        global_step=step)

  def load(self, checkpoint_dir, sess):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
      print(" [*] Successfully read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False

  def print_progress(self, step, start_step, start_time, d_loss, g_loss,
                     progress_reporter, sess):
    del sess  # Unused by abstract_gan, used for gans_with_penalty.
    if step % 100 == 0:
      time_elapsed = time.time() - start_time
      steps_per_sec = (step - start_step) / time_elapsed
      eta_seconds = (self.training_steps - step) / (steps_per_sec + 0.0000001)
      eta_minutes = eta_seconds / 60.0
      if progress_reporter:
        progress_reporter(step=step,
                          steps_per_sec=steps_per_sec,
                          progress=step/self.training_steps,
                          eta_minutes=eta_minutes)
      print("[%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f "
            "steps_per_sec: %.4f ETA: %.2f minutes" %
            (step, self.training_steps, time_elapsed, d_loss, g_loss,
             steps_per_sec, eta_minutes))

  def visualize_results(self, step, sess, z_distribution=None):
    """Generates and stores a set of fake images."""
    suffix = "%s_step%03d_test_all_classes.png" % (self.model_name, step)
    self._save_samples(step, sess, filename_suffix=suffix,
                       z_distribution=z_distribution)

  def check_variables(self, t_vars, d_vars, g_vars):
    """Make sure that every variable belongs to generator or discriminator."""
    shared_vars = set(d_vars) & set(g_vars)
    if shared_vars:
      raise ValueError("Shared trainable variables: %s" % shared_vars)
    unused_vars = set(t_vars) - set(d_vars) - set(g_vars)
    if unused_vars:
      raise ValueError("Unused trainable variables: %s" % unused_vars)

  def maybe_save_checkpoint(self, checkpoint_dir, step, sess):
    if step % self.save_checkpoint_steps == 0:
      self.save(checkpoint_dir, step, sess)
      self.visualize_results(step, sess)

  def maybe_save_samples(self, step, sess):
    """Saves training results every 5000 steps."""
    if np.mod(step, 5000) != 0:
      return
    suffix = "%s_train_%04d.png" % (self.model_name, step)
    self._save_samples(step, sess, filename_suffix=suffix)

  def _image_grid_shape(self):
    if self.batch_size & (self.batch_size - 1) != 0:
      raise ValueError("Batch size must be a power of 2 to create a grid of "
                       "fake images but was {}.".format(self.batch_size))
    # Since b = 2^c we can use x = 2^(floor(c/2)) and y = 2^(ceil(c/2)).
    x = 2 ** int(np.log2(self.batch_size) / 2)
    return x, self.batch_size // x

  def _save_samples(self, step, sess, filename_suffix, z_distribution=None):
    if z_distribution is None:
      z_distribution = self.z_generator
    z_sample = z_distribution(self.batch_size, self.z_dim)
    grid_shape = self._image_grid_shape()
    samples = sess.run(self.fake_images_merged,
                       feed_dict={self.z: z_sample})
    samples = samples.reshape((grid_shape[0] * self.input_height,
                               grid_shape[1] * self.input_width, -1)).squeeze()
    out_folder = ops.check_folder(os.path.join(self.result_dir, self.model_dir))
    full_path = os.path.join(out_folder, filename_suffix)
    ops.save_images(samples, full_path)

  def after_training_step_hook(self, sess, features, counter):
    # Called after the training step. (used by BEGAN).
    pass

  def resample_before_generator(self):
    return False

  def discriminator_feed_dict(self, features, labels):
    """Returns the feed_dict for discriminator training step."""
    del labels
    return {
        self.inputs: features["images"],
        self.z: features["z_for_disc_step"],
    }

  def generator_feed_dict(self, features, labels):
    """Returns the feed_dict for generator training step."""
    # If self.resample_before_generator() returns True
    # features["z_for_disc_step"] and features["z_for_gen_step"] will be the
    # same.
    del labels
    return {self.z: features["z_for_gen_step"]}

  def z_generator(self, batch_size, z_dim):
    """Returns the z-generator as numpy. Used during training."""
    return np.random.uniform(-1, 1, size=(batch_size, z_dim)).astype(np.float32)

  def z_tf_generator(self, batch_size, z_dim, name=None):
    """Returns the z-generator as TF op.

    Used during exported evaluation subgraph.

    Args:
      batch_size: batch size used by the graph.
      z_dim: dimensions of the z (latent) space.
      name: name for the created Tensor.

    Returns:
      Tensor object.
    """

    return tf.random_uniform(
        (batch_size, z_dim), minval=-1.0, maxval=1.0, name=name)

  def run_single_train_step(self, features, counter, g_loss, sess):
    """Runs a single training step."""

    # Update the discriminator network.
    _, summary_str, d_loss = sess.run(
        [self.d_optim, self.d_sum, self.d_loss],
        feed_dict=self.discriminator_feed_dict(features, labels=None))
    self.writer.add_summary(summary_str, counter)

    # Update the generator network.
    if (counter - 1) % self.disc_iters == 0 or g_loss is None:
      _, summary_str, g_loss = sess.run(
          [self.g_optim, self.g_sum, self.g_loss],
          feed_dict=self.generator_feed_dict(features, labels=None))
      self.writer.add_summary(summary_str, counter)

    self.after_training_step_hook(sess, features, counter)
    return d_loss, g_loss

  @contextmanager
  def _different_batch_size(self, batch_size):
    """Context to temporary change the batch size in self.batch_size.

    This is useful for TPUs where
    self.batch_size = num_tpu_cores * batch_size_per_core.
    The input pipeline will create batches of size self.batch_size but the
    subgraph for the model needs to be constructed with batch_size_per_core.

    Args:
      batch_size: Batch size to set self.batch_size for within the context.
    Yields:
      Context with self.batch_size set to batch_size.
    """
    original_batch_size = self.batch_size
    self.batch_size = batch_size
    try:
      yield
    finally:
      self.batch_size = original_batch_size

  def _input_fn(self, params):
    """Input_fn for Estimator training."""

    del params
    # If self.unroll_disc_iters is True we need to pack multiple mini-batches
    # (one per disc_iters) into one batch. Unrolling the discriminator
    # iterations is workaround until tf.cond is fully supported on TPUs.
    with self._different_batch_size(self.batch_size * self.num_sub_steps):
      dataset = self.dataset_content.batch(self.batch_size)
      dataset = dataset.prefetch(PREFETCH_NUM_BATCHES)
      images, _ = dataset.make_one_shot_iterator().get_next()
      # TPUs require known shapes everywhere.
      images = tf.to_float(tf.reshape(
          images,
          [self.batch_size, self.input_height, self.input_width, self.c_dim]))
      z = self.z_tf_generator(self.batch_size, self.z_dim,
                              name="z_for_disc_step")
      features = {
          "images": images,
          "z_for_disc_step": z,
      }
      # Some models resample the noise before training the generator. Therefore
      # our mini-batch contains noise for both discriminator and generator.
      # For models that don't resample both are the same.
      if self.resample_before_generator():
        features["z_for_gen_step"] = self.z_tf_generator(
            self.batch_size, self.z_dim, name="z_for_gen_step")
      else:
        features["z_for_gen_step"] = tf.identity(z, name="z_for_gen_step")
      return features, None

  def _model_fn(self, features, labels, mode, params):
    """Constructs the model for the given features and mode.

    Args:
      features: A dictionary with the feature tensors. See the _input_fn()
          for the set of features.
      labels: Tensor will labels. None for the implementation of _input_fn()
          above. Will be None if mode is PREDICT.
      mode: `tf.estimator.ModeKeys` value (TRAIN, EVAL, PREDICT). The mode
          should be passed to the TPUEstimatorSpec and your model should be
          build this mode.
      params: Dictionary with hyper parameters. We currently do not use this,
          but TPUEstimator will set the desired batch_size. You can use
          _different_batch_size to temporary override self.batch_size.

    Returns:
      A `tf.contrib.tpu.TPUEstimatorSpec`.
    """
    raise NotImplementedError(
        "_model_fn() must be implemented in subclasses of AbstractGAN.")

  def train_with_estimator(self, config, warm_start_from=None):
    """Trains this model using the Estimator interface.

    Args:
      config: `tf.contrib.tpu.RunConfig` for the training.
      warm_start_from: None or a `tf.estimator.WarmStartSettings` object. If
          provided variables can be initialized from a checkpoint.

    Returs:
      A `tf.contrib.tpu.TPUEstimator`.
    """
    # Noise for generating fake images.
    self.sample_z = self.z_generator(self.batch_size, self.z_dim)

    bs = self.batch_size * self.num_sub_steps
    estimator = tf.contrib.tpu.TPUEstimator(
        config=config,
        use_tpu=self.use_tpu,
        model_fn=self._model_fn,
        train_batch_size=bs,
        eval_batch_size=bs,
        warm_start_from=warm_start_from)
    estimator.train(input_fn=self._input_fn, max_steps=self.training_steps)

  def train(self, sess, progress_reporter=None):
    """Runs the training algorithm."""

    self.fake_images_merged = tf.contrib.gan.eval.image_grid(
        self.fake_images,
        grid_shape=self._image_grid_shape(),
        image_shape=(self.input_height, self.input_width),
        num_channels=self.c_dim)

    # Initialize the variables.
    global_step = tf.train.get_or_create_global_step()
    global_step_inc = global_step.assign_add(1)
    tf.global_variables_initializer().run()
    batch_size = self.batch_size

    # Noise for generating fake images.
    self.sample_z = self.z_generator(batch_size, self.z_dim)

    # Model saver and summary writer.
    self.saver = tf.train.Saver(max_to_keep=self.max_checkpoints_to_keep)
    self.writer = tf.summary.FileWriter(
        self.log_dir + "/" + self.model_name, sess.graph)

    # Restore existing checkpoints.
    could_load = self.load(self.checkpoint_dir, sess)
    if could_load:
      print(" [*] Model successfully loaded.")
    else:
      print(" [!] No checkpoint available, loading failed.")
      self.save(self.checkpoint_dir, 0, sess)

    # Start training.
    counter = tf.train.global_step(sess, global_step)
    batch_input, _ = self._input_fn(params=None)
    start_time = time.time()

    g_loss = None
    start_step = int(counter) + 1
    for step in range(start_step, self.training_steps + 1):
      # Fetch next batch and run the single training step.
      features = sess.run(batch_input)
      d_loss, g_loss = self.run_single_train_step(features, step, g_loss, sess)

      sess.run(global_step_inc)
      self.print_progress(step, start_step, start_time, d_loss, g_loss,
                          progress_reporter, sess)
      self.maybe_save_samples(step, sess)
      # Save the model and visualize current results.
      self.maybe_save_checkpoint(self.checkpoint_dir, step, sess)

    if start_step < self.training_steps:
      # Save the final model.
      self.save(self.checkpoint_dir, self.training_steps, sess)
      self.visualize_results(self.training_steps, sess)
