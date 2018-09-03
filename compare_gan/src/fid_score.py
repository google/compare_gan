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

"""Library for evaluating GAN models using Frechet Inception distance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
from six.moves import range
import tensorflow as tf

logging = tf.logging
tfgan_eval = tf.contrib.gan.eval


def get_fid_function(real_image_tensor, gen_image_tensor, num_gen_images,
                     num_eval_images, image_range, inception_graph):
  """Get a fn returning the FID between distributions defined by two tensors.

  Wraps session.run calls to generate num_eval_images images from both
  gen_image_tensor and real_image_tensor (as num_eval_images is often much
  larger than the training batch size). Then finds the FID between these two
  groups of images.

  Args:
    real_image_tensor: Tensor of shape [batch_size, dim, dim, 3] which evaluates
      to a batch of real eval images. Should be in range [0..255].
    gen_image_tensor: Tensor of shape [batch_size, dim, dim, 3] which evaluates
      to a batch of gen images. Should be in range [0..255].
    num_gen_images: Number of generated images to evaluate FID between
    num_eval_images: Number of real images to evaluate FID between
    image_range: Range of values in the images. Accepted values: "0_255".
    inception_graph: GraphDef with frozen inception model.

  Returns:
    eval_fn: a function which takes a session as an argument and returns the
      FID between num_eval_images images generated from the distributions
      defined by gen_image_tensor and real_image_tensor
  """

  assert image_range == "0_255"
  # Set up graph for generating features to pass to FID eval.
  batch_size_gen = gen_image_tensor.get_shape().as_list()[0]
  batch_size_real = real_image_tensor.get_shape().as_list()[0]

  # We want to cover only the case that the real data is bigger than
  # generated (50k vs 10k for CIFAR to be comparable with SN GAN)
  assert batch_size_real >= batch_size_gen
  assert batch_size_real % batch_size_gen == 0

  # We preprocess images and extract inception features as soon as they're
  # generated. This is to maintain memory efficiency if the images are large.
  # For example, for ImageNet, the inception features are much smaller than
  # the images.
  eval_features_tensor = get_inception_features(real_image_tensor,
                                                inception_graph)
  gen_features_tensor = get_inception_features(gen_image_tensor,
                                               inception_graph)

  num_gen_images -= num_gen_images % batch_size_gen
  num_eval_images -= num_eval_images % batch_size_real
  logging.info("Evaluating %d real images to match batch size %d",
               num_eval_images, batch_size_real)
  logging.info("Evaluating %d generated images to match batch size %d",
               num_gen_images, batch_size_gen)
  # Make sure we run the same number of batches, as this is what TFGAN code
  # assumes.
  assert num_eval_images // batch_size_real == num_gen_images // batch_size_gen
  num_batches = num_eval_images // batch_size_real

  # Set up another subgraph for calculating FID from fed images.
  feed_gen_features = tf.placeholder(
      dtype=tf.float32, shape=[num_gen_images] +
      gen_features_tensor.get_shape().as_list()[1:])
  feed_eval_features = tf.placeholder(
      dtype=tf.float32, shape=[num_eval_images] +
      eval_features_tensor.get_shape().as_list()[1:])

  # Create the tensor which stores the computed FID. We have extracted the
  # features at the point of image generation so classifier_fn=tf.identity.
  fid_tensor = tfgan_eval.frechet_classifier_distance(
      classifier_fn=tf.identity,
      real_images=feed_eval_features,
      generated_images=feed_gen_features,
      num_batches=num_batches)

  # Define a function which wraps some session.run calls to generate a large
  # number of images and compute FID on them.
  def eval_fn(session):
    """Function which wraps session.run calls to evaluate FID."""
    logging.info("Evaluating.....")
    logging.info("Generating images to feed")
    eval_features_np = []
    gen_features_np = []
    for _ in range(num_batches):
      e, g = session.run([eval_features_tensor, gen_features_tensor])
      eval_features_np.append(e)
      gen_features_np.append(g)

    logging.info("Generated images successfully.")
    eval_features_np = np.concatenate(eval_features_np)
    gen_features_np = np.concatenate(gen_features_np)

    logging.info("Computing FID with generated images...")
    fid_result = session.run(fid_tensor, feed_dict={
        feed_eval_features: eval_features_np,
        feed_gen_features: gen_features_np})

    logging.info("Computed FID: %f", fid_result)
    return fid_result

  return eval_fn


def preprocess_for_inception(images):
  """Preprocess images for inception.

  Args:
    images: images minibatch. Shape [batch size, width, height,
      channels]. Values are in [0..255].

  Returns:
    preprocessed_images
  """

  # Images should have 3 channels.
  assert images.shape[3].value == 3

  # tfgan_eval.preprocess_image function takes values in [0, 255]
  with tf.control_dependencies([tf.assert_greater_equal(images, 0.0),
                                tf.assert_less_equal(images, 255.0)]):
    images = tf.identity(images)

  preprocessed_images = tf.map_fn(
      fn=tfgan_eval.preprocess_image,
      elems=images,
      back_prop=False
  )

  return preprocessed_images


def get_inception_features(inputs, inception_graph, layer_name="pool_3:0"):
  """Compose the preprocess_for_inception function with TFGAN run_inception."""

  preprocessed = preprocess_for_inception(inputs)
  return tfgan_eval.run_inception(
      preprocessed,
      graph_def=inception_graph,
      output_tensor=layer_name)


def run_inception(images, inception_graph):
  preprocessed = tfgan_eval.preprocess_image(images)
  logits = tfgan_eval.run_inception(preprocessed, graph_def=inception_graph)
  return logits


def inception_score_fn(images, num_batches, inception_graph):
  return tfgan_eval.classifier_score(
      images, num_batches=num_batches,
      classifier_fn=functools.partial(run_inception,
                                      inception_graph=inception_graph))
