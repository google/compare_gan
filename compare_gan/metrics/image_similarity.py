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

"""Image Similarity Metrics in TensorFlow.

This file contains various image similarity metrics that can be
used for monitoring how various models learn to reconstruct images
or may be used directly as the final objective to be optimized.


How to use these metrics:
  * MS-SSIM: For GANS, MS-SSIM can be used to detect mode collapse by looking at
    the similarity between samples and comparing it with the similarity inside
    the dataset. Often, this is done for class conditional models, where per
    class samples are being compared. In the unconditional setting, it can be
    done for datasets where the data has the same modality (
    for example, CelebA). For more information, see the following papers:

        * https://arxiv.org/abs/1610.09585
        * https://arxiv.org/abs/1706.04987
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from six.moves import zip
import tensorflow as tf


def verify_compatible_shapes(img1, img2):
  """Checks if two image tensors are compatible for metric computation.

  This function checks if two sets of images have ranks at least 3, and if the
  last three dimensions match.

  Args:
    img1: The first images tensor.
    img2: The second images tensor.

  Returns:
    A tuple of the first tensor shape, the second tensor shape, and a list of
    tf.Assert() implementing the checks.

  Raises:
    ValueError: when static shape check fails.
  """
  shape1 = img1.get_shape().with_rank_at_least(3)
  shape2 = img2.get_shape().with_rank_at_least(3)
  shape1[-3:].assert_is_compatible_with(shape2[-3:])

  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(reversed(shape1[:-3]), reversed(shape2[:-3])):
      if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
        raise ValueError(
            'Two images are not compatible: %s and %s' % (shape1, shape2))

  # Now assign shape tensors.
  shape1, shape2 = tf.shape_n([img1, img2])

  checks = []
  checks.append(tf.Assert(tf.greater_equal(tf.size(shape1), 3),
                          [shape1, shape2], summarize=10))
  checks.append(tf.Assert(tf.reduce_all(tf.equal(shape1[-3:], shape2[-3:])),
                          [shape1, shape2], summarize=10))
  return shape1, shape2, checks


_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def _ssim_helper(x, y, reducer, max_val, compensation=1.0):
  r"""Helper function to SSIM.

  SSIM estimates covariances with weighted sums, e.g., normalized Gaussian blur.
  Like the unbiased covariance estimator has normalization factor of n-1 instead
  of n, naive covariance estimations with weighted sums are biased estimators.
  Suppose `reducer` is a weighted sum, then the mean estimators are
    mu_x = \sum_i w_i x_i,
    mu_y = \sum_i w_i y_i,
  where w_i's are the weighted-sum weights, and covariance estimator is
    cov_xy = \sum_i w_i (x_i - mu_x) (y_i - mu_y)
  with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    E cov_xy = (1 - \sum_i w_i ** 2) Cov(X, Y).
  For SSIM measure with unbiased covariance estimators, pass as `compensation`
  argument (1 - \sum_i w_i ** 2).

  Arguments:
    x: first set of images.
    y: first set of images.
    reducer: Function that computes 'local' averages from set of images.
      For non-covolutional version, this is usually tf.reduce_mean(x, [1, 2]),
      and for convolutional version, this is usually tf.nn.avg_pool or
      tf.nn.conv2d with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.

  Returns:
    A pair containing the luminance measure and the contrast-structure measure.
  """
  c1 = (_SSIM_K1 * max_val) ** 2
  c2 = (_SSIM_K2 * max_val) ** 2

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = tf.square(mean0) + tf.square(mean1)
  luminance = (num0 + c1) / (den0 + c1)

  # SSIM contrast-structure measure is
  #   (2 * cov_xy + c2) / (cov_xx + cov_yy + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_xy = \sum_i w_i (x_i - mu_x) (y_i - mu_y)
  #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
  num1 = reducer(x * y) * 2.0
  den1 = reducer(tf.square(x) + tf.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs


def f_special_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = tf.convert_to_tensor(size, tf.int32)
  sigma = tf.convert_to_tensor(sigma)

  coords = tf.cast(tf.range(size), sigma.dtype)
  coords -= tf.cast(size - 1, sigma.dtype) / 2.0

  g = tf.square(coords)
  g *= -0.5 / tf.square(sigma)

  g = tf.reshape(g, shape=[1, -1]) + tf.reshape(g, shape=[-1, 1])
  g = tf.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = tf.nn.softmax(g)
  return tf.reshape(g, shape=[size, size, 1, 1])


def _ssim_index_per_channel(
    img1, img2, filter_size, filter_width, max_val=255.0):
  """Computes SSIM index between img1 and img2 per color channel.

  This function matches the standard SSIM implementation found at:
  https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m

  Details:
    - To reproduce a 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  Args:
    img1: First RGB image batch.
    img2: Second RGB image batch.
    filter_size: An integer, the filter size of the Gaussian kernel used.
    filter_width: A float, the filter width of the Gaussian kernel used.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).

  Returns:
    A pair of tensors containing batch-wise and channel-wise SSIM and
    contrast-structure measure. The shape is [..., channels].
  """
  filter_size = tf.constant(filter_size, dtype=tf.int32)
  filter_sigma = tf.constant(filter_width, dtype=img1.dtype)

  shape1, shape2 = tf.shape_n([img1, img2])

  filter_size = tf.reduce_min(
      tf.concat([tf.expand_dims(filter_size, axis=0),
                 shape1[-3:-1],
                 shape2[-3:-1]],
                axis=0))

  kernel = f_special_gauss(filter_size, filter_sigma)
  kernel = tf.tile(kernel, multiples=[1, 1, shape1[-1], 1])

  # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
  # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
  compensation = 1.0


  def reducer(x):  # pylint: disable=invalid-name
    shape = tf.shape(x)
    x = tf.reshape(x, shape=tf.concat([[-1], shape[-3:]], 0))
    y = tf.nn.depthwise_conv2d(x, kernel, strides=[1] * 4, padding='VALID')
    return tf.reshape(y, tf.concat([shape[:-3], tf.shape(y)[1:]], 0))

  luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation)

  # Average over the second and the third from the last: height, width.
  axes = tf.constant([-3, -2], dtype=tf.int32)
  ssim = tf.reduce_mean(luminance * cs, axes)
  cs = tf.reduce_mean(cs, axes)
  return ssim, cs


# This must be a tuple (not a list) because tuples are immutable and we don't
# want these to accidentally change.
_MSSSIM_WEIGHTS = (.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def multiscale_ssim(
    img1, img2, filter_size=11, filter_width=1.5, max_val=255.0):
  """Computes MS-SSIM with power factors from Wang paper."""
  return _multiscale_ssim_helper(img1, img2,
                                 filter_size=filter_size,
                                 filter_width=filter_width,
                                 max_val=max_val,
                                 power_factors=_MSSSIM_WEIGHTS)


def multiscale_ssim_unweighted(
    img1, img2, filter_size=11, filter_width=1.5, max_val=255.0):
  """Computes unweighted MS-SSIM with power factors from Zhao paper."""
  return _multiscale_ssim_helper(img1, img2,
                                 filter_size=filter_size,
                                 filter_width=filter_width,
                                 max_val=max_val,
                                 power_factors=[1, 1, 1, 1, 1])


def _multiscale_ssim_helper(
    img1, img2, filter_size, filter_width, power_factors, max_val=255.0):
  """Computes the MS-SSIM between img1 and img2.

  This function assumes that `img1` and `img2` are image batches, i.e. the last
  three dimensions are [row, col, channels].

  Arguments:
    img1: First RGB image batch.
    img2: Second RGB image batch. Must have the same rank as img1.
    filter_size: An integer, the filter size of the Gaussian kernel used.
    filter_width: A float, the filter width of the Gaussian kernel used.
    power_factors: iterable of weightings for each of the scales. The number of
      scales used is the length of the list. Index 0 is the unscaled
      resolution's weighting and each increasing scale corresponds to the image
      being downsampled by 2.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).

  Returns:
    A tensor containing batch-wise MS-SSIM measure. MS-SSIM has range [0, 1].
    The shape is broadcast(img1.shape[:-3], img2.shape[:-3]).
  """
  # Shape checking.
  shape1 = img1.get_shape().with_rank_at_least(3)
  shape2 = img2.get_shape().with_rank_at_least(3)
  shape1[-3:].merge_with(shape2[-3:])

  with tf.name_scope(None, 'MS-SSIM', [img1, img2]):
    shape1, shape2, checks = verify_compatible_shapes(img1, img2)
    with tf.control_dependencies(checks):
      img1 = tf.identity(img1)

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]

    divisor = [1, 2, 2, 1]
    divisor_tensor = tf.constant(divisor[1:], dtype=tf.int32)

    def do_pad(images, remainder):  # pylint: disable=invalid-name
      padding = tf.expand_dims(remainder, -1)
      padding = tf.pad(padding, [[1, 0], [1, 0]])
      return [tf.pad(x, padding, mode='SYMMETRIC') for x in images]

    mcs = []
    for k in range(len(power_factors)):
      with tf.name_scope(None, 'Scale%d' % k, imgs):
        if k > 0:
          # Avg pool takes rank 4 tensors. Flatten leading dimensions.
          flat_imgs = [
              tf.reshape(x, tf.concat([[-1], t], 0))
              for x, t in zip(imgs, tails)
          ]

          remainder = tails[0] % divisor_tensor
          need_padding = tf.reduce_any(tf.not_equal(remainder, 0))
          # pylint: disable=cell-var-from-loop
          padded = tf.cond(need_padding,
                           lambda: do_pad(flat_imgs, remainder),
                           lambda: flat_imgs)
          # pylint: enable=cell-var-from-loop

          downscaled = [
              tf.nn.avg_pool(
                  x, ksize=divisor, strides=divisor, padding='VALID')
              for x in padded
          ]
          tails = [x[1:] for x in tf.shape_n(downscaled)]
          imgs = [
              tf.reshape(x, tf.concat([h, t], 0))
              for x, h, t in zip(downscaled, heads, tails)
          ]

        # Overwrite previous ssim value since we only need the last one.
        ssim, cs = _ssim_index_per_channel(
            *imgs,
            filter_size=filter_size, filter_width=filter_width,
            max_val=max_val)
        mcs.append(tf.nn.relu(cs))

    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    mcs_and_ssim = tf.stack(mcs + [tf.nn.relu(ssim)], axis=-1)
    # Take weighted geometric mean across the scale axis.
    ms_ssim = tf.reduce_prod(tf.pow(mcs_and_ssim, power_factors), [-1])

    ms_ssim = tf.reduce_mean(ms_ssim, [-1])  # Average over color channels.
    return ms_ssim
