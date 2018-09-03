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

"""Implementation of GANs (MMGAN, NSGAN, WGAN) with different regularizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src.gans import ablation_resnet_architecture
from compare_gan.src.gans import consts
from compare_gan.src.gans import dcgan_architecture
from compare_gan.src.gans import resnet_architecture
from compare_gan.src.gans.abstract_gan import AbstractGAN

from six.moves import range
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("print_penalty_loss", False,
                     "Whether to print penalty loss.")


class AbstractGANWithPenalty(AbstractGAN):
  """GAN class for which we can modify loss/penalty/architecture via params."""

  def __init__(self, model_name, **kwargs):
    super(AbstractGANWithPenalty, self).__init__(model_name, **kwargs)

    self.architecture = self.parameters.get("architecture")
    self.penalty_type = self.parameters.get("penalty_type")
    self.discriminator_normalization = self.parameters.get(
        "discriminator_normalization")

    if self.penalty_type not in consts.PENALTIES:
      raise ValueError("Penalty '%s' not recognized." % self.penalty_type)

    if self.discriminator_normalization not in consts.NORMALIZERS:
      raise ValueError("Normalization '%s' not recognized." % (
          self.discriminator_normalization))

    if self.architecture not in consts.ARCHITECTURES:
      raise ValueError("Architecture '%s' not recognized." % self.architecture)

    # Regularization strength.
    self.lambd = self.parameters["lambda"]
    self.beta2 = self.parameters.get("beta2", 0.999)

    # If the optimizer wasn't specified, use Adam to be consistent with
    # other GANs.
    self.optimizer = self.parameters.get("optimizer", "adam")

    self.resnet_ablation_type = self.parameters.get(
        "resnet_ablation_type", "none")

  def get_optimizer(self, name_prefix):
    if self.optimizer == "adam":
      print("Using Adam optimizer.")
      return tf.train.AdamOptimizer(
          self.learning_rate,
          beta1=self.beta1,
          beta2=self.beta2,
          name=name_prefix + self.optimizer)
    elif self.optimizer == "rmsprop":
      print("Using RMSProp optimizer.")
      return tf.train.RMSPropOptimizer(
          self.learning_rate, name=name_prefix + self.optimizer)
    elif self.optimizer == "sgd":
      print("Using GradientDescent optimizer.")
      return tf.train.GradientDescentOptimizer(self.learning_rate)
    else:
      raise ValueError("Unknown optimizer: %s" % self.optimizer)

  def print_progress(self, step, start_step, start_time, d_loss, g_loss,
                     progress_reporter, sess):
    super(AbstractGANWithPenalty, self).print_progress(
        step, start_step, start_time, d_loss, g_loss, progress_reporter, sess)
    if FLAGS.print_penalty_loss and step % 100 == 0:
      penalty_loss = sess.run(self.penalty_loss)
      print("\t\tlambda: %.4f penalty_loss: %.4f" % (self.lambd, penalty_loss))

  def discriminator(self, x, is_training, reuse=False):
    if self.architecture == consts.INFOGAN_ARCH:
      return super(AbstractGANWithPenalty, self).discriminator(
          x, is_training, reuse)
    elif self.architecture == consts.DCGAN_ARCH:
      return dcgan_architecture.discriminator(
          x, self.batch_size, is_training,
          self.discriminator_normalization, reuse)
    elif self.architecture == consts.RESNET5_ARCH:
      return resnet_architecture.resnet5_discriminator(
          x, is_training, self.discriminator_normalization, reuse)
    elif self.architecture == consts.RESNET107_ARCH:
      return resnet_architecture.resnet107_discriminator(
          x, is_training, self.discriminator_normalization, reuse)
    elif self.architecture == consts.RESNET_CIFAR:
      return resnet_architecture.resnet_cifar_discriminator(
          x, is_training, self.discriminator_normalization, reuse)
    elif self.architecture == consts.RESNET_STL:
      return resnet_architecture.resnet_stl_discriminator(
          x, is_training, self.discriminator_normalization, reuse)
    elif self.architecture == consts.SNDCGAN_ARCH:
      assert self.discriminator_normalization in [
          consts.SPECTRAL_NORM, consts.NO_NORMALIZATION]
      return dcgan_architecture.sn_discriminator(
          x, self.batch_size, reuse,
          use_sn=self.discriminator_normalization == consts.SPECTRAL_NORM)
    elif self.architecture == consts.RESNET5_ABLATION:
      return ablation_resnet_architecture.resnet5_discriminator(
          x, is_training, self.discriminator_normalization, reuse,
          unused_ablation_type=self.resnet_ablation_type)
    else:
      raise NotImplementedError(
          "Architecture %s not implemented." % self.architecture)

  def generator(self, z, is_training, reuse=False):
    if self.architecture == consts.INFOGAN_ARCH:
      return super(AbstractGANWithPenalty, self).generator(
          z, is_training, reuse)
    elif self.architecture == consts.DCGAN_ARCH:
      return dcgan_architecture.generator(z, self.batch_size,
                                          self.output_height, self.output_width,
                                          self.c_dim, is_training, reuse)
    elif self.architecture == consts.RESNET5_ARCH:
      assert self.output_height == self.output_width
      return resnet_architecture.resnet5_generator(
          z,
          is_training=is_training,
          reuse=reuse,
          colors=self.c_dim,
          output_shape=self.output_height)
    elif self.architecture == consts.RESNET_STL:
      return resnet_architecture.resnet_stl_generator(
          z, is_training=is_training, reuse=reuse, colors=self.c_dim)
    elif self.architecture == consts.RESNET107_ARCH:
      return resnet_architecture.resnet107_generator(
          z, is_training=is_training, reuse=reuse, colors=self.c_dim)
    elif self.architecture == consts.RESNET_CIFAR:
      return resnet_architecture.resnet_cifar_generator(
          z, is_training=is_training, reuse=reuse, colors=self.c_dim)
    elif self.architecture == consts.SNDCGAN_ARCH:
      return dcgan_architecture.sn_generator(
          z, self.batch_size, self.output_height, self.output_width, self.c_dim,
          is_training, reuse)
    elif self.architecture == consts.RESNET5_ABLATION:
      assert self.output_height == self.output_width
      return ablation_resnet_architecture.resnet5_generator(
          z, is_training=is_training, reuse=reuse, colors=self.c_dim,
          output_shape=self.output_height,
          unused_ablation_type=self.resnet_ablation_type)
    else:
      raise NotImplementedError(
          "Architecture %s not implemented." % self.architecture)

  def dragan_penalty(self, x, discriminator, is_training):
    """Returns the DRAGAN gradient penalty."""
    _, var = tf.nn.moments(x, axes=list(range(len(x.get_shape()))))
    std = tf.sqrt(var)
    x_noisy = x + std * (tf.random_uniform(x.shape) - 0.5)
    x_noisy = tf.clip_by_value(x_noisy, 0.0, 1.0)
    logits = discriminator(x_noisy, is_training=is_training, reuse=True)[1]
    gradients = tf.gradients(logits, [x_noisy])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
    return gradient_penalty

  def wgangp_penalty(self, x, x_fake, discriminator, is_training):
    """Returns the WGAN gradient penalty."""
    alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1])
    interpolates = x + alpha * (x_fake - x)
    logits = discriminator(interpolates, is_training=is_training, reuse=True)[1]
    gradients = tf.gradients(logits, [interpolates])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
    return gradient_penalty

  def l2_penalty(self):
    """Returns the L2 penalty for each matrix/vector excluding biases."""
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    # Linear + conv2d and deconv2d layers.
    print(d_vars)
    d_weights = [
        v for v in d_vars
        if ((v.name.endswith("/Matrix:0") and "fc" in v.name) or
            (v.name.endswith("/w:0") and "conv" in v.name))
    ]
    if len(d_weights) != consts.N_DISCRIMINATOR_LAYERS[self.architecture]:
      raise RuntimeError(
          "l2_penalty: got %d layers(%s), expected %d layers for %s." %
          (len(d_weights), d_weights,
           consts.N_DISCRIMINATOR_LAYERS[self.architecture], self.architecture))
    return tf.reduce_mean(
        [tf.nn.l2_loss(i) for i in d_weights], name="l2_penalty")

  def _model_fn(self, features, labels, mode, params):
    del labels  # unused.
    tf.logging.info("model_fn(): features=%s, mode=%s, params=%s)", features,
                    mode, params)

    if mode == tf.estimator.ModeKeys.PREDICT:

      noise = features["z"]
      predictions = {
          "generated_images": self.generator(noise, is_training=False)
      }
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    global_step = tf.train.get_or_create_global_step()

    def create_sub_step_loss(features, sub_step_idx=0, reuse=True):
      """Creates the loss for a slice of the current batch.

      Args:
        features: A dictionary with the feature tensors.
        sub_step_idx: Index of the slice of the batch to use to construct the
            loss. If self.unroll_disc_iters is True this must be 0 and the whole
            batch will be used.
        reuse: Bool, whether to reuse existing variables for the models.
            Should be False for the first call and True on all other calls.
      """

      tf.logging.info("sub_step_idx: %s, params: %s, unroll_disc_iters: %s",
                      sub_step_idx, params, self.unroll_disc_iters)
      assert (sub_step_idx == 0) or self.unroll_disc_iters
      bs = params["batch_size"] // self.num_sub_steps
      with self._different_batch_size(bs):
        begin = sub_step_idx * self.batch_size
        end = begin + self.batch_size
        f = {k: v[begin:end] for k, v in features.iteritems()}
        self.create_loss(f, is_training=is_training, reuse=reuse)
        self.fake_images = self.generator(f["z"], is_training=False, reuse=True)

    if mode == tf.estimator.ModeKeys.TRAIN:
      #########
      # TRAIN #
      #########
      # Discriminator training.
      # Create the model and loss for z_for_disc_step features.
      features["z"] = features["z_for_disc_step"]
      create_sub_step_loss(features, reuse=False)

      # Divide trainable variables into a group for D and group for G.
      t_vars = tf.trainable_variables()
      d_vars = [var for var in t_vars if "discriminator" in var.name]
      g_vars = [var for var in t_vars if "generator" in var.name]
      self.check_variables(t_vars, d_vars, g_vars)

      # Discriminator training.
      deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(deps):
        d_optimizer = self.get_optimizer("d_")
        g_optimizer = self.get_optimizer("g_")
        if self.use_tpu:
          d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)
          g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)

        deps.append(
            d_optimizer.minimize(
                self.d_loss, var_list=d_vars, global_step=global_step))

      if self.unroll_disc_iters:
        for sub_step in range(1, self.disc_iters):
          with tf.control_dependencies(deps):
            create_sub_step_loss(features, sub_step)
            deps.append(
                d_optimizer.minimize(
                    self.d_loss, var_list=d_vars, global_step=global_step))

      # Generator training.
      # Create loss using the z_for_gen_step features.
      features["z"] = features["z_for_gen_step"]
      if self.num_sub_steps > 1:
        create_sub_step_loss(features, self.num_sub_steps - 1)
      else:
        create_sub_step_loss(features)

      with tf.control_dependencies(deps):
        if self.unroll_disc_iters or self.disc_iters == 1:
          train_op = g_optimizer.minimize(self.g_loss, var_list=g_vars)
        else:
          # We should only train the generator every self.disc_iter steps.
          # We can do this using `tf.cond`. Both paths must return a tensor.
          # Our true_fn will return a tensor that depends on training the
          # generator, while the tensor from false_fn depends on nothing.
          def do_train_generator():
            actual_train_op = g_optimizer.minimize(self.g_loss, var_list=g_vars)
            with tf.control_dependencies([actual_train_op]):
              return tf.constant(0)
          def do_not_train_generator():
            return tf.constant(0)
          counter = tf.to_int32(global_step) - 1
          train_op = tf.cond(
              tf.equal(counter % self.disc_iters, 0),
              true_fn=do_train_generator,
              false_fn=do_not_train_generator).op
    else:
      train_op = None

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        # Estimator requires a loss which gets displayed on TensorBoard.
        # The given Tensor is evaluated but not used to create gradients.
        loss=self.d_loss,
        train_op=train_op)

  def create_loss(self, features, is_training=True, reuse=False):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss, self.g_loss and self.penalty_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).
      reuse: Bool, whether to reuse existing variables for the models.
          This is only used for unrolling discriminator iterations when training
          on TPU.

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    images = features["images"]  # Input images.
    z = features["z"]  # Noise vector.

    # Discriminator output for real images.
    d_real, d_real_logits, _ = self.discriminator(
        images, is_training=is_training, reuse=reuse)

    # Discriminator output for fake images.
    generated = self.generator(z, is_training=is_training, reuse=reuse)
    d_fake, d_fake_logits, _ = self.discriminator(
        generated, is_training=is_training, reuse=True)

    self.discriminator_output = d_real

    if self.model_name not in consts.MODELS_WITH_PENALTIES:
      raise ValueError("Model %s not recognized" % self.model_name)

    # Define the loss functions
    if self.model_name == consts.GAN_WITH_PENALTY:
      d_loss_real = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=d_real_logits, labels=tf.ones_like(d_real)))
      d_loss_fake = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=d_fake_logits, labels=tf.zeros_like(d_fake)))
      self.d_loss = d_loss_real + d_loss_fake
      self.g_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=d_fake_logits, labels=tf.ones_like(d_fake)))
    elif self.model_name == consts.WGAN_WITH_PENALTY:
      d_loss_real = -tf.reduce_mean(d_real_logits)
      d_loss_fake = tf.reduce_mean(d_fake_logits)
      self.d_loss = d_loss_real + d_loss_fake
      self.g_loss = -d_loss_fake
    elif self.model_name == consts.LSGAN_WITH_PENALTY:
      d_loss_real = tf.reduce_mean(tf.square(d_real - 1.0))
      d_loss_fake = tf.reduce_mean(tf.square(d_fake))
      self.d_loss = 0.5 * (d_loss_real + d_loss_fake)
      self.g_loss = 0.5 * tf.reduce_mean(tf.square(d_fake - 1.0))
    elif self.model_name == consts.SN_GAN_WITH_PENALTY:
      d_loss_real = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
      d_loss_fake = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
      self.d_loss = d_loss_real + d_loss_fake
      self.g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake_logits))
    else:
      raise ValueError("Unknown GAN model_name: %s" % self.model_name)

    # Define the penalty.
    if self.penalty_type == consts.NO_PENALTY:
      self.penalty_loss = 0.0
    elif self.penalty_type == consts.DRAGAN_PENALTY:
      self.penalty_loss = self.dragan_penalty(images, self.discriminator,
                                              is_training)
      self.d_loss += self.lambd * self.penalty_loss
    elif self.penalty_type == consts.WGANGP_PENALTY:
      self.penalty_loss = self.wgangp_penalty(images, generated,
                                              self.discriminator, is_training)
      self.d_loss += self.lambd * self.penalty_loss
    elif self.penalty_type == consts.L2_PENALTY:
      self.penalty_loss = self.l2_penalty()
      self.d_loss += self.lambd * self.penalty_loss
    else:
      raise NotImplementedError(
          "The penalty %s was not implemented." % self.penalty_type)

    # Setup summaries.

    if not self.use_tpu:
      d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
      d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
      d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
      g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
      self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
      self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

  def build_model(self, is_training=True):
    """Build the model (input placeholders, losses and optimizer ops).

    Args:
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).
    """
    image_dims = [self.input_height, self.input_width, self.c_dim]
    batch_size = self.batch_size
    # Input images.
    self.inputs = tf.placeholder(
        tf.float32, [batch_size] + image_dims, name="real_images")
    # Noise vector.
    self.z = tf.placeholder(tf.float32, [batch_size, self.z_dim], name="z")

    features = {"images": self.inputs, "z": self.z}
    self.create_loss(features, is_training=is_training)

    # Store testing images.
    self.fake_images = self.generator(self.z, is_training=False, reuse=True)

    # This subgraph will be used to evaluate generators.
    # It doesn't require any inputs and just keeps generating images.
    with tf.name_scope("generator_evaluation"):
      eval_input = self.z_tf_generator(
          self.batch_size, self.z_dim, name="input")
      result = self.generator(eval_input, is_training=False, reuse=True)
      result = tf.identity(result, name="result")

    # This subraph can be used as representation.
    # It pins the current values of batch_norm, and has a separate input tensor.
    with tf.name_scope("discriminator_representation"):
      result, _, _ = self.discriminator(
          tf.placeholder(tf.float32, self.inputs.get_shape(), name="input"),
          is_training=False,
          reuse=True)
      result = tf.identity(result, name="result")

    # Divide trainable variables into a group for D and group for G.
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    g_vars = [var for var in t_vars if "generator" in var.name]
    self.check_variables(t_vars, d_vars, g_vars)

    # Define optimization ops.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      self.d_optim = self.get_optimizer("d_").minimize(
          self.d_loss, var_list=d_vars)
      self.g_optim = self.get_optimizer("g_").minimize(
          self.g_loss, var_list=g_vars)


class GAN_PENALTY(AbstractGANWithPenalty):
  """Non-saturating Generative Adverserial Networks.

     The loss for the generator is computed using the log trick. That is,
     G_loss = -log(D(fake_images))  [maximizes log(D)]
  """

  def __init__(self, **kwargs):
    super(GAN_PENALTY, self).__init__(consts.GAN_WITH_PENALTY, **kwargs)


class WGAN_PENALTY(AbstractGANWithPenalty):
  """Generative Adverserial Networks with the Wasserstein loss."""

  def __init__(self, **kwargs):
    super(WGAN_PENALTY, self).__init__(consts.WGAN_WITH_PENALTY, **kwargs)


class LSGAN_PENALTY(AbstractGANWithPenalty):
  """Generative Adverserial Networks with the Least Squares loss."""

  def __init__(self, **kwargs):
    super(LSGAN_PENALTY, self).__init__(consts.LSGAN_WITH_PENALTY, **kwargs)
