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

import functools
import itertools

from absl import flags
from absl import logging
from compare_gan import utils
from compare_gan.architectures import abstract_arch
from compare_gan.architectures import dcgan
from compare_gan.architectures import infogan
from compare_gan.architectures import resnet30
from compare_gan.architectures import resnet5
from compare_gan.architectures import resnet5_biggan
from compare_gan.architectures import resnet_cifar
from compare_gan.architectures import resnet_stl
from compare_gan.architectures import sndcgan
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans import penalty_lib
from compare_gan.gans.abstract_gan import AbstractGAN
from compare_gan.tpu import tpu_random
from compare_gan.tpu import tpu_summaries
import gin
import numpy as np
from six.moves import range
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

FLAGS = flags.FLAGS


@gin.configurable(blacklist=["dataset", "parameters", "model_dir"])
class ModularGAN(AbstractGAN):
  """Base class for GANs models that support the Estimator API."""

  def __init__(self,
               dataset,
               parameters,
               model_dir,
               deprecated_split_disc_calls=False,
               experimental_joint_gen_for_disc=False,
               g_use_ema=False,
               ema_decay=0.9999,
               ema_start_step=40000,
               g_optimizer_fn=tf.train.AdamOptimizer,
               d_optimizer_fn=None,
               g_lr=0.0002,
               d_lr=None,
               conditional=False,
               fit_label_distribution=False):
    """ModularGAN  is a Gin configurable implementation of AbstractGAN.

    Args:
      dataset: `ImageDataset` object. If `conditional` the dataset must provide
        labels and the number of classes bust known.
      parameters: Legacy Python dictionary with additional parameters. This must
        have the keys 'architecture', 'z_dim' and 'lambda'.
      model_dir: Directory path for storing summary files.
      deprecated_split_disc_calls: If True pass fake and real images separately
        through the discriminator network.
      experimental_joint_gen_for_disc: If True generate fake images for all D
        iterations jointly. This increase the batch size in G when generating
        fake images for D. The G step is stays the same.
      g_use_ema: If True keep moving averages for weights in G and use them in
        the TF-Hub module.
      ema_decay: Decay rate for moving averages for G's weights.
      ema_start_step: Start step for keeping moving averages. Before this the
        decay rate is 0.
      g_optimizer_fn: Function (or constructor) to return an optimizer for G.
      d_optimizer_fn: Function (or constructor) to return an optimizer for D.
        If None will call `g_optimizer_fn`.
      g_lr: Learning rate for G.
      d_lr: Learning rate for D. Defaults to `g_lr`.
      conditional: Whether the GAN is conditional. If True both G and Y will
        get passed labels.
      fit_label_distribution: Whether to fit the label distribution.
    """
    super(ModularGAN, self).__init__(
        dataset=dataset, parameters=parameters, model_dir=model_dir)
    self._deprecated_split_disc_calls = deprecated_split_disc_calls
    self._experimental_joint_gen_for_disc = experimental_joint_gen_for_disc
    self._g_use_ema = g_use_ema
    self._ema_decay = ema_decay
    self._ema_start_step = ema_start_step
    self._g_optimizer_fn = g_optimizer_fn
    self._d_optimizer_fn = d_optimizer_fn
    if self._d_optimizer_fn is None:
      self._d_optimizer_fn = g_optimizer_fn
    self._g_lr = g_lr
    self._d_lr = g_lr if d_lr is None else d_lr

    if conditional and not self._dataset.num_classes:
      raise ValueError(
          "Option 'conditional' selected but dataset {} does not have "
          "labels".format(self._dataset.name))
    self._conditional = conditional
    self._fit_label_distribution = fit_label_distribution

    self._tpu_summary = tpu_summaries.TpuSummaries(model_dir)

    # Parameters that have not been ported to Gin.
    self._architecture = parameters["architecture"]
    self._z_dim = parameters["z_dim"]
    self._lambda = parameters["lambda"]

    # Number of discriminator iterations per one iteration of the generator.
    self._disc_iters = parameters.get("disc_iters", 1)

    # Will be set by create_loss().
    self.d_loss = None
    self.g_loss = None
    self.penalty_loss = None

  @property
  def num_sub_steps(self):
    # We are training D and G separately. Since TPU require use to have a single
    # training operations we join multiple sub steps in one step. One sub step
    # for each discriminator training step (disc_iters) and a separate sub step
    # (with new inputs) for the generator training step.
    return self._disc_iters + 1

  @property
  def conditional(self):
    return self._conditional

  def as_estimator(self, run_config, batch_size, use_tpu):
    """Returns a TPUEstimator for this GAN."""
    return tf.contrib.tpu.TPUEstimator(
        config=run_config,
        use_tpu=use_tpu,
        model_fn=self.model_fn,
        train_batch_size=batch_size * self.num_sub_steps)

  def _module_fn(self, model, batch_size):
    """Module Function to create a TF Hub module spec.

    Args:
      model: `tf.estimator.ModeKeys` value.
      batch_size: batch size.
    """
    if model not in {"gen", "disc"}:
      raise ValueError("Model {} not support in module_fn()".format(model))
    placeholder_fn = tf.placeholder if batch_size is None else tf.zeros
    is_training = False
    inputs = {}
    y = None
    if model == "gen":
      inputs["z"] = placeholder_fn(
          shape=(batch_size, self._z_dim),
          dtype=tf.float32,
          name="z_for_eval")
    elif model == "disc":
      inputs["images"] = placeholder_fn(
          shape=[batch_size] + list(self._dataset.image_shape),
          dtype=tf.float32,
          name="images_for_eval")
    if self.conditional:
      inputs["labels"] = placeholder_fn(
          shape=(batch_size,),
          dtype=tf.int32,
          name="labels_for_eval")
      y = self._get_one_hot_labels(inputs["labels"])
    else:
      y = None

    logging.info("Creating module for model %s with inputs %s and y=%s",
                 model, inputs, y)
    outputs = {}
    if model == "disc":
      outputs["prediction"], _, _ = self.discriminator(
          inputs["images"], y=y, is_training=is_training)
    else:
      z = inputs["z"]
      generated = self.generator(z, y=y, is_training=is_training)
      if self._g_use_ema and not is_training:
        g_vars = [var for var in tf.trainable_variables()
                  if "generator" in var.name]
        ema = tf.train.ExponentialMovingAverage(decay=self._ema_decay)
        # Create the variables that will be loaded from the checkpoint.
        ema.apply(g_vars)
        def ema_getter(getter, name, *args, **kwargs):
          var = getter(name, *args, **kwargs)
          ema_var = ema.average(var)
          if ema_var is None:
            var_names_without_ema = {"u_var", "accu_mean", "accu_variance",
                                     "accu_counter", "update_accus"}
            if name.split("/")[-1] not in var_names_without_ema:
              logging.warning("Could not find EMA variable for %s.", name)
            return var
          return ema_var
        with tf.variable_scope("", values=[z, y], reuse=True,
                               custom_getter=ema_getter):
          generated = self.generator(z, y=y, is_training=is_training)
      outputs["generated"] = generated

    hub.add_signature(inputs=inputs, outputs=outputs)

  def as_module_spec(self):
    """Returns the generator network as TFHub module spec."""
    models = ["gen", "disc"]
    batch_sizes = [8, 16, 32, 64, 128]
    if "resnet" in self._architecture:
      # Only ResNet architectures support dynamic batch size.
      batch_sizes.append(None)
    tags_and_args = [(set(), {"model": "gen", "batch_size": 64})]
    for model, bs in itertools.product(models, batch_sizes):
      tags = {model, "bs{}".format(bs)}
      args = {"model": model, "batch_size": bs}
      tags_and_args.append((tags, args))
    return hub.create_module_spec(
        self._module_fn, tags_and_args=tags_and_args,
        drop_collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES])

  def _samples_and_grid_shape(self, max_samples_per_replica, num_replicas):
    total_samples = num_replicas
    while total_samples < 64 or total_samples & (total_samples - 1) != 0:
      total_samples += num_replicas
    samples_per_replica = min(max_samples_per_replica,
                              total_samples // num_replicas)
    num_summary_images = samples_per_replica * num_replicas
    if num_summary_images & (num_summary_images - 1) != 0:
      raise ValueError(
          "Number of summary images must be a power of 2 to create a grid of "
          "images but was {}.".format(num_summary_images))
    # Since b = 2^c we can use x = 2^(floor(c/2)) and y = 2^(ceil(c/2)).
    x = 2 ** int(np.log2(num_summary_images) / 2)
    y = num_summary_images // x
    return samples_per_replica, (x, y)

  def _add_images_to_summary(self, images, summary_name, params):
    """Called from model_fn() to add a grid of images as summary."""
    num_replicas = params["context"].num_replicas if "context" in params else 1
    max_samples_per_replica = params["batch_size"] // self.num_sub_steps
    samples_per_replica, grid_shape = (
        self._samples_and_grid_shape(max_samples_per_replica, num_replicas))
    def _merge_images_to_grid(tensor):
      logging.info("Creating images summary for fake images: %s", tensor)
      return tfgan.eval.image_grid(
          tensor,
          grid_shape=grid_shape,
          image_shape=self._dataset.image_shape[:2],
          num_channels=self._dataset.image_shape[2])
    self._tpu_summary.image(summary_name,
                            images[:samples_per_replica],
                            reduce_fn=_merge_images_to_grid)

  def _check_variables(self, t_vars, d_vars, g_vars):
    """Make sure that every variable belongs to generator or discriminator."""
    shared_vars = set(d_vars) & set(g_vars)
    if shared_vars:
      raise ValueError("Shared trainable variables: %s" % shared_vars)
    unused_vars = set(t_vars) - set(d_vars) - set(g_vars)
    if unused_vars:
      raise ValueError("Unused trainable variables: %s" % unused_vars)

  def _get_one_hot_labels(self, labels):
    if not self.conditional:
      raise ValueError(
          "_get_one_hot_labels() called but GAN is not conditional.")
    return tf.one_hot(labels, self._dataset.num_classes)

  @gin.configurable("z", blacklist=["shape", "name"])
  def z_generator(self, shape, distribution_fn=tf.random.uniform,
                  minval=-1.0, maxval=1.0, stddev=1.0, name=None):
    """Random noise distributions as TF op.

    Args:
      shape: A 1-D integer Tensor or Python array.
      distribution_fn: Function that create a Tensor. If the function has any
        of the arguments 'minval', 'maxval' or 'stddev' these are passed to it.
      minval: The lower bound on the range of random values to generate.
      maxval: The upper bound on the range of random values to generate.
      stddev: The standard deviation of a normal distribution.
      name: A name for the operation.

    Returns:
      Tensor with the given shape and dtype tf.float32.
    """
    return utils.call_with_accepted_args(
        distribution_fn, shape=shape, minval=minval, maxval=maxval,
        stddev=stddev, name=name)

  def label_generator(self, shape, name=None):
    if not self.conditional:
      raise ValueError("label_generator() called but GAN is not conditional.")
    # Assume uniform label distribution.
    return tf.random.uniform(shape, minval=0, maxval=self._dataset.num_classes,
                             dtype=tf.int32, name=name)

  def _preprocess_fn(self, images, labels, seed=None):
    """Creates the feature dictionary with images and z."""
    logging.info("_preprocess_fn(): images=%s, labels=%s, seed=%s",
                 images, labels, seed)
    tf.set_random_seed(seed)
    features = {
        "images": images,
        "z": self.z_generator([self._z_dim], name="z"),
    }
    if self.conditional:
      if self._fit_label_distribution:
        features["sampled_labels"] = labels
      else:
        features["sampled_labels"] = self.label_generator(
            shape=[], name="sampled_labels")
    return features, labels

  def input_fn(self, params, mode):
    """Input function that retuns a `tf.data.Dataset` object.

    This function will be called once for each host machine.

    Args:
      params: Python dictionary with parameters given to TPUEstimator.
          Additional TPUEstimator will set the key `batch_size` with the batch
          size for this host machine and `tpu_contextu` with a TPUContext
          object.
      mode: `tf.estimator.MoedeKeys` value.

    Returns:
      A `tf.data.Dataset` object with batched features and labels.
    """
    return self._dataset.input_fn(mode=mode, params=params,
                                  preprocess_fn=self._preprocess_fn)

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

    def _create_sub_step_loss(sub_step_idx=0, reuse=True):
      """Creates the loss for a slice of the current batch.

      Args:
        sub_step_idx: Index of the slice of the batch to use to construct the
            loss. If self.unroll_disc_iters is True this must be 0 and the whole
            batch will be used.
        reuse: Bool, whether to reuse existing variables for the models.
            Should be False for the first call and True on all other calls.

      Returns:
        Fake images created by the generator.
      """
      logging.info("sub_step_idx: %s, params: %s", sub_step_idx, params)
      # Set the random offset tensor for operations in tpu_random.py.
      tpu_random.set_random_offset_from_features(fs[sub_step_idx])
      self.create_loss(fs[sub_step_idx], ls[sub_step_idx], params,
                       is_training=is_training, reuse=reuse)

    # Split inputs for sub steps.
    fs = [(k, tf.split(features[k], self.num_sub_steps)) for k in features]
    fs = [{k: v[i] for k, v in fs} for i in range(self.num_sub_steps)]
    ls = tf.split(labels, self.num_sub_steps)

    # Only the last sub step changes the generator weights. Thus we can combine
    # all forward passes through G to achieve better efficiency. The forward
    # pass for G's step needs to be separated since compute gradients for it.
    if self._experimental_joint_gen_for_disc:
      logging.info("Running generator forward pass for all D steps.")
      with tf.name_scope("gen_for_disc"):
        bs = params["batch_size"] // self.num_sub_steps
        # D steps.
        z = features["z"][:-bs]
        sampled_y = None
        if self.conditional:
          sampled_y = self._get_one_hot_labels(features["sampled_labels"][:-bs])
        generated = tf.stop_gradient(self.generator(
            z, y=sampled_y, is_training=is_training, reuse=False))
        assert self.num_sub_steps - 1 == self._disc_iters
        generated = tf.split(generated, self._disc_iters)
        for i in range(self._disc_iters):
          fs[i]["generated"] = generated[i]
          del fs[i]["z"]
        # G step.
        z = features["z"][-bs:]
        sampled_y = None
        if self.conditional:
          sampled_y = self._get_one_hot_labels(features["sampled_labels"][-bs:])
        fs[-1]["generated"] = self.generator(
            z, y=sampled_y, is_training=is_training, reuse=True)
        del fs[-1]["z"]

    logging.info("fs=%s, ls=%s", fs, ls)
    # Create ops for first D steps here to create the variables.
    with tf.name_scope("disc_step_1"):
      _create_sub_step_loss(0, reuse=tf.AUTO_REUSE)
      d_losses = [self.d_loss]

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
          self.d_loss, var_list=d_vars))

    for sub_step_idx in range(1, self._disc_iters):
      with tf.name_scope("disc_step_{}".format(sub_step_idx + 1)):
        with tf.control_dependencies(deps):
          _create_sub_step_loss(sub_step_idx)
          deps.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        with tf.control_dependencies(deps):
          d_losses.append(self.d_loss)
          deps.append(d_optimizer.minimize(
              self.d_loss, var_list=d_vars))

    # Clean old summaries from previous calls to model_fn().
    self._tpu_summary = tpu_summaries.TpuSummaries(self._model_dir)
    for i, d_loss in enumerate(d_losses):
      self._tpu_summary.scalar("loss/d_{}".format(i), d_loss)
    if self._experimental_joint_gen_for_disc:
      fake_images = fs[0]["generated"]
    else:
      with tf.name_scope("fake_images"):
        z = fs[0]["z"]
        sampled_y = None
        if self.conditional:
          sampled_y = self._get_one_hot_labels(fs[0]["sampled_labels"])
        fake_images = self.generator(
            z, y=sampled_y, is_training=True, reuse=True)
    self._add_images_to_summary(fake_images, "fake_images", params)
    self._add_images_to_summary(fs[0]["images"], "real_images", params)

    # Generator training.
    with tf.name_scope("gen_step"):
      with tf.control_dependencies(deps):
        # This will use the same inputs as the last for the discriminator step
        # above, but the new sub-graph will depend on the updates of the
        # discriminator steps.
        _create_sub_step_loss(self.num_sub_steps - 1)
        self._tpu_summary.scalar("loss/g", self.g_loss)
        deps.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
      with tf.control_dependencies(deps):
        train_op = g_optimizer.minimize(self.g_loss, var_list=g_vars,
                                        global_step=global_step)
        loss = self.g_loss

    if self._g_use_ema:
      with tf.name_scope("generator_ema"):
        logging.info("Creating moving averages of weights: %s", g_vars)
        # The decay value is set to 0 if we're before the moving-average start
        # point, so that the EMA vars will be the normal vars.
        decay = self._ema_decay * tf.cast(
            tf.greater_equal(global_step, self._ema_start_step), tf.float32)
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        with tf.control_dependencies([train_op]):
          train_op = ema.apply(g_vars)

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

  def generator(self, z, y, is_training, reuse=False):
    """Returns the generator network."""
    architecture_fns = {
        c.DCGAN_ARCH: dcgan.Generator,
        c.INFOGAN_ARCH: infogan.Generator,
        c.RESNET5_BIGGAN_ARCH: resnet5_biggan.Generator,
        c.RESNET5_ARCH: resnet5.Generator,
        c.RESNET30_ARCH: resnet30.Generator,
        c.RESNET_STL: resnet_stl.Generator,
        c.RESNET_CIFAR: resnet_cifar.Generator,
        c.SNDCGAN_ARCH: sndcgan.Generator,
    }
    if self._architecture not in architecture_fns:
      raise NotImplementedError(
          "Architecture {} not implemented.".format(self._architecture))
    generator = architecture_fns[self._architecture](
        image_shape=self._dataset.image_shape)
    assert isinstance(generator, abstract_arch.AbstractGenerator)

    # Functionality to learn the label distribution.
    if self.conditional and self._fit_label_distribution:
      with tf.variable_scope("label_counts", reuse=reuse):
        label_counts = tf.get_variable(
            "label_counts",
            initializer=tf.constant_initializer(1e-6),
            shape=self._dataset.num_classes,
            dtype=tf.float32,
            trainable=False)
      num_active_labels = tf.reduce_sum(tf.cast(tf.greater(label_counts, .5),
                                                tf.float32))
      self._tpu_summary.scalar("label_counts/active", num_active_labels)
      if is_training:
        logging.info("Learning the label distribution by counting.")
        new_counts = tf.contrib.tpu.cross_replica_sum(tf.reduce_sum(y, axis=0))
        update_label_counts_op = tf.assign_add(label_counts, new_counts)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_label_counts_op)
      else:
        logging.info("Sampling labels using the label counts.")
        normalized_counts = label_counts / tf.reduce_sum(label_counts)
        dist = tf.distributions.Categorical(probs=normalized_counts)
        sampled_labels = dist.sample(tf.shape(y)[0])
        with tf.control_dependencies([y]):
          y = self._get_one_hot_labels(sampled_labels)
    return generator(z=z, y=y, is_training=is_training, reuse=reuse)

  def discriminator(self, x, y, is_training, reuse=False):
    """Returns the discriminator network."""
    architecture_fns = {
        c.DCGAN_ARCH: dcgan.Discriminator,
        c.INFOGAN_ARCH: infogan.Discriminator,
        c.RESNET5_ARCH: resnet5.Discriminator,
        c.RESNET5_BIGGAN_ARCH: resnet5_biggan.Discriminator,
        c.RESNET30_ARCH: resnet30.Discriminator,
        c.RESNET_STL: resnet_stl.Discriminator,
        c.RESNET_CIFAR: resnet_cifar.Discriminator,
        c.SNDCGAN_ARCH: sndcgan.Discriminator,
    }
    if self._architecture not in architecture_fns:
      raise NotImplementedError(
          "Architecture {} not implemented.".format(self._architecture))
    discriminator = architecture_fns[self._architecture]()
    assert isinstance(discriminator, abstract_arch.AbstractDiscriminator)
    return discriminator(x=x, y=y, is_training=is_training, reuse=reuse)

  def d_optimizer(self, use_tpu=True):
    opt = self._d_optimizer_fn(self._d_lr, name="d_")
    if use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    return opt

  def g_optimizer(self, use_tpu=True):
    opt = self._g_optimizer_fn(self._g_lr, name="g_")
    if use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    return opt

  def create_loss(self, features, labels, params, is_training=True,
                  reuse=False):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss and self.g_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      labels: Tensor will labels. Use
          self._get_one_hot_labels(labels) to get a one hot encoded tensor.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).
      reuse: Bool, whether to reuse existing variables for the models.
          This is only used for unrolling discriminator iterations when training
          on TPU.

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    images = features["images"]  # Input images.
    if self.conditional:
      y = self._get_one_hot_labels(labels)
      sampled_y = self._get_one_hot_labels(features["sampled_labels"])
      all_y = tf.concat([y, sampled_y], axis=0)
    else:
      y = None
      sampled_y = None
      all_y = None

    if self._experimental_joint_gen_for_disc:
      assert "generated" in features
      generated = features["generated"]
    else:
      logging.warning("Computing fake images for sub step separately.")
      z = features["z"]  # Noise vector.
      generated = self.generator(
          z, y=sampled_y, is_training=is_training, reuse=reuse)

    if self._deprecated_split_disc_calls:
      with tf.name_scope("disc_for_real"):
        d_real, d_real_logits, _ = self.discriminator(
            images, y=y, is_training=is_training, reuse=reuse)
      with tf.name_scope("disc_for_fake"):
        d_fake, d_fake_logits, _ = self.discriminator(
            generated, y=sampled_y, is_training=is_training, reuse=True)
    else:
      # Compute discriminator output for real and fake images in one batch.
      all_images = tf.concat([images, generated], axis=0)
      d_all, d_all_logits, _ = self.discriminator(
          all_images, y=all_y, is_training=is_training, reuse=reuse)
      d_real, d_fake = tf.split(d_all, 2)
      d_real_logits, d_fake_logits = tf.split(d_all_logits, 2)

    self.d_loss, _, _, self.g_loss = loss_lib.get_losses(
        d_real=d_real, d_fake=d_fake, d_real_logits=d_real_logits,
        d_fake_logits=d_fake_logits)

    discriminator = functools.partial(self.discriminator, y=y)
    penalty_loss = penalty_lib.get_penalty_loss(
        x=images, x_fake=generated, is_training=is_training,
        discriminator=discriminator, architecture=self._architecture)
    self.d_loss += self._lambda * penalty_loss
    self._tpu_summary.scalar("loss/penalty", penalty_loss)
