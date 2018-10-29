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

"""Evaluation for GAN tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from compare_gan.src import gan_lib
from compare_gan.src import params
from compare_gan.src import simple_task_pb2
from compare_gan.src import task_utils
from compare_gan.src.gans import consts
from compare_gan.src.multi_gan import dataset
from compare_gan.src.multi_gan import multi_gan
from compare_gan.src.multi_gan import multi_gan_background

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

flags = tf.flags

flags.DEFINE_string("out_dir", None, "Where to save images.")
flags.DEFINE_string("eval_task_workdir", None, "Workdir to evaluate")

flags.DEFINE_string("checkpoint", "all",
                    "Which checkpoint(s) to evaluate for a given study/task."
                    "Supports {'all', <int>}.")
flags.DEFINE_enum("visualization_type", "multi_image", [
    "image", "multi_image", "latent", "multi_latent"],
                  "How to visualize this GAN.")
flags.DEFINE_integer("batch_size", 64, "Size of the batch.")
flags.DEFINE_integer("images_per_fig", 4, "How many images to stack in a fig.")
FLAGS = flags.FLAGS


def GetBackgroundImageTensorName(architecture):
  """Returns the name of the tensor that is the generated background."""

  tensor_prefix = "background_generator/generator"
  if architecture == consts.DCGAN_ARCH:
    tensor_suffix = "add:0"
  elif architecture == consts.RESNET5_ARCH:
    tensor_suffix = "Sigmoid:0"
  else:
    raise ValueError("Unknown architecture: %s" % architecture)

  return "%s/%s" % (tensor_prefix, tensor_suffix)


def PlotImage(ax, im, title=None, x_label=None, y_label=None, vmin=0.0,
              vmax=1.0, cmap="gray"):
  """Helper function that wraps matplotlib.axes.Axes.imshow.

  Args:
    ax: An instance of matplotlib.axes.Axes
    im: The image to plot (W, H, C)
    title: The title of the image
    x_label: Optional x label
    y_label: Optional y label
    vmin: The minimum value that a pixel can take.
    vmax: The maximum value that a pixel can take.
    cmap: The color map to use (whenever C != 3)

  """

  im = im[..., 0] if im.shape[2] == 1 else im
  ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)

  # Set xlabel
  if x_label: ax.set_xlabel(x_label, fontsize=16)
  if y_label: ax.set_ylabel(y_label, rotation=90, fontsize=16)

  # Remove axis.
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xticklabels([])
  ax.set_yticklabels([])

  # Set title.
  if title is not None: ax.set_title(title)


def SaveGanLatentTraversalImages(aggregated_images, save_dir):
  """Visualizes latent traversals.

  Args:
    aggregated_images: The aggregated image (B, T, W, H, C).
    save_dir: The path to the directory in which the figure should be saved.
  """

  def LatentTraversalFig(aggregated_images_):
    """Returns a figure that visualizes a sequence of latent traversals."""
    n_images, n_steps, _, _, _ = aggregated_images_.shape

    figsize = (n_steps * 5, n_images * 5)
    fig, ax = plt.subplots(n_images, n_steps, figsize=figsize)

    for i in range(n_images):
      for t in range(n_steps):
        x_label = "step %d" % t if i == n_images - 1 else None
        y_label = "Generated %d" % i if t == 0 else None
        ax_ = ax[i, t] if n_images > 1 else ax[t]
        PlotImage(ax_, aggregated_images_[i, t],
                  x_label=x_label, y_label=y_label)

    return fig

  images_per_fig = FLAGS.images_per_fig
  for block in range(0, aggregated_images.shape[0], images_per_fig):
    figure = LatentTraversalFig(
        aggregated_images[block:block + images_per_fig])
    plt.subplots_adjust(wspace=0.1, hspace=0.04)
    plt.savefig(os.path.join(
        save_dir, "single_latent_interpolation_%d-%d.pdf" % (
            block, min(block + images_per_fig, aggregated_images.shape[0]))),
                bbox_inches="tight")
    plt.close(figure)


def SaveMultiGanLatentTraversalImages(aggregated_images, generated_images,
                                      save_dir):
  """Visualizes latent traversals factored across generators.

  Args:
    aggregated_images: The aggregated image (B, T, W, H, C).
    generated_images: The output of each generator (B, T, K, W, H, C)
    save_dir: The path to the directory in which the figure should be saved.
  """

  def LatentTraversalFig(aggregated_images_, generated_images_):
    """Returns a figure that visualizes a sequence of latent traversals."""
    n_images, n_steps, k, _, _, _ = generated_images_.shape

    figsize = (n_steps * 5, (k + 1) * n_images * 5)
    fig, ax = plt.subplots((k + 1)* n_images, n_steps, figsize=figsize)

    for i in range(0, (k + 1) * n_images, (k + 1)):
      for t in range(n_steps):
        PlotImage(ax[i, t], aggregated_images_[i // (k + 1), t])

        for j in range(k):
          PlotImage(ax[i + j + 1, t], generated_images_[i // (k + 1), t, j])

    return fig

  images_per_fig = FLAGS.images_per_fig
  for block in range(0, aggregated_images.shape[0], images_per_fig):
    figure = LatentTraversalFig(
        aggregated_images[block:block + images_per_fig],
        generated_images[block:block + images_per_fig])
    plt.savefig(os.path.join(
        save_dir, "multi_latent_interpolation_%d-%d.pdf" % (
            block, min(block + images_per_fig, aggregated_images.shape[0]))))
    plt.close(figure)


def SaveMultiGanGeneratorImages(aggregated_images, generated_images, save_dir):
  """Visualizes the output of each generator together with its aggregate.

  Args:
    aggregated_images: The aggregated image (B, W, H, C).
    generated_images: The output of each generator (B, K, W, H, C)
    save_dir: The path to the directory in which the figure should be saved.
  """

  images_per_fig = FLAGS.images_per_fig
  for block in range(0, aggregated_images.shape[0], images_per_fig):
    b_aggregated_images = aggregated_images[block:block + images_per_fig]
    b_generated_images = generated_images[block:block + images_per_fig]
    n_images, k, _, _, _ = b_generated_images.shape

    figsize = ((k + 1) * 5, n_images * 5)
    fig, ax = plt.subplots(n_images, (k + 1), figsize=figsize)

    for i in range(n_images):
      for j in range(k):
        ax_ = ax[i, j] if n_images > 1 else ax[j]
        PlotImage(
            ax_, b_generated_images[i, k - j - 1], x_label="Generator %d" % j)

      ax_ = ax[i, -1] if n_images > 1 else ax[-1]
      PlotImage(ax_, b_aggregated_images[i], x_label="Composed Image")

    plt.subplots_adjust(wspace=0.1, hspace=0.04)
    plt.savefig(os.path.join(save_dir, "generator_images_%d-%d.pdf" % (
        block, min(block + images_per_fig, aggregated_images.shape[0]))),
                bbox_inches="tight")
    plt.close(fig)


def SaveGeneratorImages(aggregated_images, save_dir):
  """Visualizes the aggregated output of all generators.

  Args:
    aggregated_images: The aggregated image (B, W, H, C).
    save_dir: The path to the directory in which the figure should be saved.
  """

  n_images, _, _, _ = aggregated_images.shape

  for i in range(n_images):
    fig, ax = plt.subplots(figsize=(5, 5))

    PlotImage(ax, aggregated_images[i])
    plt.savefig(os.path.join(save_dir, "generator_images_%d.png" % i),
                bbox_inches="tight")
    plt.close(fig)


def GetMultiGANGeneratorsOp(graph, gan_type, architecture, aggregate):
  """Returns the op to obtain the output of all generators."""

  # Generator ops for normal Multi-GAN.
  generator_preds_op = graph.get_tensor_by_name("generator_predictions:0")

  # Add background generator
  if gan_type == "MultiGANBackground":
    background_image_name = GetBackgroundImageTensorName(architecture)
    generator_preds_bg_op = graph.get_tensor_by_name(background_image_name)

    # Set background alpha to 1. as is done pre-aggregation during training
    if aggregate == "alpha":
      generator_preds_bg_op = tf.concat(
          (generator_preds_bg_op[..., :-1], tf.ones_like(
              generator_preds_bg_op[..., -1:])), axis=-1)

    generator_preds_op = tf.concat(
        [generator_preds_op, tf.expand_dims(
            generator_preds_bg_op, axis=1)], axis=1)

  return generator_preds_op


def EvalCheckpoint(checkpoint_path, task_workdir, options, out_cp_dir):
  """Evaluate model at given checkpoint_path."""

  # Overwrite batch size
  options["batch_size"] = FLAGS.batch_size

  checkpoint_dir = os.path.join(task_workdir, "checkpoint")
  result_dir = os.path.join(task_workdir, "result")
  gan_log_dir = os.path.join(task_workdir, "logs")

  dataset_params = params.GetDatasetParameters(options["dataset"])
  dataset_params.update(options)

  # generate fake images
  with tf.Graph().as_default() as g:
    with tf.Session() as sess:
      gan = gan_lib.create_gan(
          gan_type=options["gan_type"],
          dataset=options["dataset"],
          dataset_content=None,
          options=options,
          checkpoint_dir=checkpoint_dir,
          result_dir=result_dir,
          gan_log_dir=gan_log_dir)

      gan.build_model(is_training=False)

      tf.global_variables_initializer().run()
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)

      # Compute outputs for MultiGanGeneratorImages.
      if (FLAGS.visualization_type == "multi_image" and
          "MultiGAN" in options["gan_type"]):
        generator_preds_op = GetMultiGANGeneratorsOp(
            g, options["gan_type"], options["architecture"],
            options["aggregate"])

        fetches = [gan.fake_images, generator_preds_op]

        # Construct feed dict
        z_sample = gan.z_generator(gan.batch_size, gan.z_dim)
        feed_dict = {gan.z: z_sample}

        # Fetch data and save images.
        fake_images, generator_preds = sess.run(fetches, feed_dict=feed_dict)
        SaveMultiGanGeneratorImages(fake_images, generator_preds, out_cp_dir)

      # Compute outputs for GeneratorImages.
      elif FLAGS.visualization_type == "image":
        # Construct feed dict
        z_sample = gan.z_generator(gan.batch_size, gan.z_dim)
        feed_dict = {gan.z: z_sample}

        # Fetch data and save images.
        fake_images = sess.run(gan.fake_images, feed_dict=feed_dict)
        SaveGeneratorImages(fake_images, out_cp_dir)

      # Compute outputs for MultiGanLatentTraversalImages
      elif (FLAGS.visualization_type == "multi_latent" and
            "MultiGAN" in options["gan_type"]):
        generator_preds_op = GetMultiGANGeneratorsOp(
            g, options["gan_type"], options["architecture"],
            options["aggregate"])

        fetches = [gan.fake_images, generator_preds_op]

        # Init latent params
        z_sample = gan.z_generator(gan.batch_size, gan.z_dim)
        directions = np.random.uniform(size=z_sample.shape)
        k_indices = np.random.randint(gan.k, size=gan.batch_size)
        n_steps, step_size = 10, 0.1
        images, gen_preds = [], []

        # Traverse in latent space of a single component n_steps times and
        # generate the corresponding images.
        for step in range(n_steps + 1):
          new_z = z_sample.copy()
          for i in range(z_sample.shape[0]):
            new_z[i, k_indices[i]] += (
                step * step_size * directions[i, k_indices[i]])

          images_batch, gen_preds_batch = sess.run(fetches, {gan.z: new_z})
          images.append(images_batch)
          gen_preds.append(gen_preds_batch)

        images = np.stack(images, axis=1)
        gen_preds = np.stack(gen_preds, axis=1)
        SaveMultiGanLatentTraversalImages(images, gen_preds, out_cp_dir)

      # Compute outputs for GanLatentTraversalImages
      elif FLAGS.visualization_type == "latent":
        # Init latent params.
        z_sample = gan.z_generator(gan.batch_size, gan.z_dim)
        directions = np.random.uniform(size=z_sample.shape)
        k_indices = np.random.randint(options.get("k", 1), size=gan.batch_size)
        n_steps, step_size = 5, 0.1
        images = []

        # Traverse in latent space of a single component n_steps times and
        # generate the corresponding images.
        for step in range(n_steps + 1):
          new_z = z_sample.copy()
          for i in range(z_sample.shape[0]):
            if "MultiGAN" in options["gan_type"]:
              new_z[i, k_indices[i]] += (
                  step * step_size * directions[i, k_indices[i]])
            else:
              new_z[i] += step * step_size * directions[i]

          images_batch = sess.run(gan.fake_images, {gan.z: new_z})
          images.append(images_batch)

        images = np.stack(images, axis=1)
        SaveGanLatentTraversalImages(images, out_cp_dir)


def GetModelDir(options):
  """Returns the model directory for a given model."""

  model_dir = ""
  if options["gan_type"] in ["MultiGAN", "MultiGANBackground"]:
    if options["n_blocks"] == 0 and options["n_heads"] == 0:
      model_dir = "Independent"
    model_dir += "%s-%d" % (options["gan_type"], options["k"])
  else:
    model_dir = "GAN"

  return model_dir


def EvalTask(options, task_workdir, out_dir):
  """Evaluates all checkpoints for the given task."""

  # Compute all records not done yet.
  checkpoint_dir = os.path.join(task_workdir, "checkpoint")
  # Fetch checkpoint to eval.
  checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)

  if FLAGS.checkpoint == "all":
    all_checkpoint_paths = checkpoint_state.all_model_checkpoint_paths
  else:
    all_checkpoint_paths = ["%s/checkpoint/%s.model-%s" % (
        FLAGS.eval_task_workdir, options["gan_type"], FLAGS.checkpoint)]

  for checkpoint_path in all_checkpoint_paths:
    out_cp_dir = os.path.join(
        out_dir, "checkpoint-%s" % checkpoint_path.split("-")[-1])
    if not tf.gfile.IsDirectory(out_cp_dir):
      tf.gfile.MakeDirs(out_cp_dir)
    EvalCheckpoint(checkpoint_path, task_workdir, options, out_cp_dir)


def main(unused_argv):
  gan_lib.MODELS.update({
      "MultiGAN": multi_gan.MultiGAN,
      "MultiGANBackground": multi_gan_background.MultiGANBackground
  })
  params.PARAMETERS.update({
      "MultiGAN": multi_gan.MultiGANHyperParams,
      "MultiGANBackground": multi_gan_background.MultiGANBackgroundHyperParams
  })

  gan_lib.DATASETS.update(dataset.get_datasets())
  params.DATASET_PARAMS.update(dataset.get_dataset_params())

  task_workdir = FLAGS.eval_task_workdir

  task = simple_task_pb2.Task()
  with open(os.path.join(task_workdir, "task"), "r") as f:
    text_format.Parse(f.read(), task)

  options = task_utils.ParseOptions(task)

  out_dir = os.path.join(FLAGS.out_dir, GetModelDir(options))
  if not tf.gfile.IsDirectory(out_dir):
    tf.gfile.MakeDirs(out_dir)

  task_string = text_format.MessageToString(task)
  print("\nWill evaluate task\n%s\n\n", task_string)

  EvalTask(options, task_workdir, out_dir)

if __name__ == "__main__":
  flags.mark_flag_as_required("eval_task_workdir")
  flags.mark_flag_as_required("out_dir")
  tf.app.run()
