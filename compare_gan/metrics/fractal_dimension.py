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

"""Implementation of the fractal dimension metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.metrics import eval_task

import numpy as np
import scipy.spatial


class FractalDimensionTask(eval_task.EvalTask):
  """Fractal dimension metric."""

  _LABEL = "fractal_dimension"

  def run_after_session(self, options, eval_data_fake, eval_data_real=None):
    print(eval_data_fake)
    score = compute_fractal_dimension(eval_data_fake.images)
    return {self._LABEL: score}


def compute_fractal_dimension(fake_images,
                              num_fd_seeds=100,
                              n_bins=1000,
                              scale=0.1):
  """Compute Fractal Dimension of fake_images.

  Args:
    fake_images: an np array of datapoints, the dimensionality and scaling of
      images can be arbitrary
    num_fd_seeds: number of random centers from which fractal dimension
      computation is performed
     n_bins: number of bins to split the range of distance values into
     scale: the scale of the y interval in the log-log plot for which we apply a
       linear regression fit

  Returns:
    fractal dimension of the dataset.
  """
  assert len(fake_images.shape) >= 2
  assert fake_images.shape[0] >= num_fd_seeds

  num_images = fake_images.shape[0]
  # In order to apply scipy function we need to flatten the number of dimensions
  # to 2
  fake_images = np.reshape(fake_images, (num_images, -1))
  fake_images_subset = fake_images[np.random.randint(
      num_images, size=num_fd_seeds)]

  distances = scipy.spatial.distance.cdist(fake_images,
                                           fake_images_subset).flatten()
  min_distance = np.min(distances[np.nonzero(distances)])
  max_distance = np.max(distances)
  buckets = min_distance * (
      (max_distance / min_distance)**np.linspace(0, 1, n_bins))
  # Create a table where first column corresponds to distances r
  # and second column corresponds to number of points N(r) that lie
  # within distance r from the random seeds
  fd_result = np.zeros((n_bins - 1, 2))
  fd_result[:, 0] = buckets[1:]
  fd_result[:, 1] = np.sum(np.less.outer(distances, buckets[1:]), axis=0)

  # We compute the slope of the log-log plot at the middle y value
  # which is stored in y_val; the linear regression fit is computed on
  # the part of the plot that corresponds to an interval around y_val
  # whose size is 2*scale*(total width of the y axis)
  max_y = np.log(num_images * num_fd_seeds)
  min_y = np.log(num_fd_seeds)
  x = np.log(fd_result[:, 0])
  y = np.log(fd_result[:, 1])
  y_width = max_y - min_y
  y_val = min_y + 0.5 * y_width

  start = np.argmax(y > y_val - scale * y_width)
  end = np.argmax(y > y_val + scale * y_width)

  slope = np.linalg.lstsq(
      a=np.vstack([x[start:end], np.ones(end - start)]).transpose(),
      b=y[start:end].reshape(end - start, 1))[0][0][0]
  return slope
