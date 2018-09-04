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

"""Library for evaluating GANs using KID score."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from compare_gan.src import fid_score
import numpy as np
import tensorflow as tf

logging = tf.logging


def get_kid_function(real_image_tensor,
                     gen_image_tensor,
                     num_gen_images,
                     num_eval_images,
                     image_range,
                     inception_graph,
                     kid_max_batch_size=1024):
  """Returns the function that computes KID score."""
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
  eval_features_tensor = fid_score.get_inception_features(
      real_image_tensor, inception_graph)
  gen_features_tensor = fid_score.get_inception_features(
      gen_image_tensor, inception_graph)

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

  feed_gen_features = tf.placeholder(
      dtype=tf.float32,
      shape=[num_gen_images] + gen_features_tensor.get_shape().as_list()[1:])
  feed_eval_features = tf.placeholder(
      dtype=tf.float32,
      shape=[num_eval_images] + eval_features_tensor.get_shape().as_list()[1:])

  # Create the tensor which stores the computed KID.
  kid_tensor = kid(
      feed_eval_features, feed_gen_features, max_batch_size=kid_max_batch_size)

  # Define a function which wraps some session.run calls to generate a large
  # number of images and compute FID on them.
  def eval_fn(session):
    """Function which wraps session.run calls to evaluate kid."""

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

    logging.info("Computing KID with generated images...")
    kid_result = session.run(
        kid_tensor,
        feed_dict={
            feed_eval_features: eval_features_np,
            feed_gen_features: gen_features_np
        })
    logging.info("Computed KID: %f", kid_result)

    return kid_result

  return eval_fn


def kid(real_activations,
        generated_activations,
        max_batch_size=1024,
        dtype=None,
        return_stderr=False):
  """Unbiased estimator of the Kernel Inception Distance.

  As defined by https://arxiv.org/abs/1801.01401.

  If return_stderr, also returns an estimate of the standard error, i.e. the
  standard deviation of the KID estimator. Returns nan if the number
  of batches is too small (< 5); for more reliable estimates, one could use
  the asymptotic variance estimate given in https://arxiv.org/abs/1611.04488.

  Uses a block estimator, as in https://arxiv.org/abs/1307.1954, with blocks
  no larger than max_batch_size. This is slightly different than the authors'
  provided code, but is also unbiased (and provides more-valid a variance
  estimate).

  NOTE: the blocking code assumes that real_activations and
  generated_activations are in random order. If real_activations is sorted
  in a meaningful order, the estimator will be biased.

  Args:
    real_activations: [batch, num_features] tensor with inception features.
    generated_activations: [batch, num_features] tensor with inception features.
    max_batch_size: Batches to compute the KID.
    dtype: Type used by the computations.
    return_stderr: If true, also returns the std_error from the KID computation.

  Returns:
    kid score (and optionally std error).
  """
  real_activations.get_shape().assert_has_rank(2)
  generated_activations.get_shape().assert_has_rank(2)

  # need to know dimension for the kernel, and batch size to split things
  real_activations.get_shape().assert_is_fully_defined()
  generated_activations.get_shape().assert_is_fully_defined()

  n_real, dim = real_activations.get_shape().as_list()
  n_gen, dim2 = generated_activations.get_shape().as_list()
  assert dim2 == dim

  # tfgan forces doubles for FID, but I don't think we need that here
  if dtype is None:
    dtype = real_activations.dtype
    assert generated_activations.dtype == dtype
  else:
    real_activations = tf.cast(real_activations, dtype)
    generated_activations = tf.cast(generated_activations, dtype)

  # split into largest approximately-equally-sized blocks
  n_bins = int(math.ceil(max(n_real, n_gen) / max_batch_size))
  bins_r = np.full(n_bins, int(math.ceil(n_real / n_bins)))
  bins_g = np.full(n_bins, int(math.ceil(n_gen / n_bins)))
  bins_r[:(n_bins * bins_r[0]) - n_real] -= 1
  bins_g[:(n_bins * bins_r[0]) - n_gen] -= 1
  assert bins_r.min() >= 2
  assert bins_g.min() >= 2

  inds_r = tf.constant(np.r_[0, np.cumsum(bins_r)])
  inds_g = tf.constant(np.r_[0, np.cumsum(bins_g)])

  dim_ = tf.cast(dim, dtype)

  def get_kid_batch(i):
    """Computes KID on a given batch of features.

    Takes real_activations[ind_r[i] : ind_r[i+1]] and
    generated_activations[ind_g[i] : ind_g[i+1]].

    Args:
     i: is the index of the batch.

    Returns:
      KID for the given batch.
    """
    r_s = inds_r[i]
    r_e = inds_r[i + 1]
    r = real_activations[r_s:r_e]
    m = tf.cast(r_e - r_s, dtype)

    g_s = inds_g[i]
    g_e = inds_g[i + 1]
    g = generated_activations[g_s:g_e]
    n = tf.cast(r_e - r_s, dtype)

    # Could probably do this a bit faster...
    k_rr = (tf.matmul(r, r, transpose_b=True) / dim_ + 1)**3
    k_rg = (tf.matmul(r, g, transpose_b=True) / dim_ + 1)**3
    k_gg = (tf.matmul(g, g, transpose_b=True) / dim_ + 1)**3
    return (
        -2 * tf.reduce_mean(k_rg) + (tf.reduce_sum(k_rr) - tf.trace(k_rr)) /
        (m * (m - 1)) + (tf.reduce_sum(k_gg) - tf.trace(k_gg)) / (n * (n - 1)))

  ests = tf.map_fn(
      get_kid_batch, np.arange(n_bins), dtype=dtype, back_prop=False)

  if return_stderr:
    if n_bins < 5:
      return tf.reduce_mean(ests), np.nan
    mn, var = tf.nn.moments(ests, [0])
    return mn, tf.sqrt(var / n_bins)
  else:
    return tf.reduce_mean(ests)
