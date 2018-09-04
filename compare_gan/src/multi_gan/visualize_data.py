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

"""Simple script to visualize samples of a custom dataset in datasets.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from compare_gan.src.multi_gan import dataset
import numpy as np
from scipy.misc import imsave
import tensorflow as tf

flags = tf.flags

flags.DEFINE_string("dataset", "multi-mnist-3-uniform", "Name of the dataset.")
flags.DEFINE_string("dataset_split", "test", "Name of the split.")
flags.DEFINE_string("save_dir", "/tmp/vis", "Directory of where to save ims.")
flags.DEFINE_boolean("grid", False, "Whether to aggregate the images in a grid "
                                    "(8 x 8) or save them in individual figs.")

FLAGS = flags.FLAGS


def save_image(im, name):
  im = im[:, :, 0] if im.shape[2] == 1 else im
  imsave(os.path.join(FLAGS.save_dir, name), im)


def main(unused_argv):
  if not tf.gfile.IsDirectory(FLAGS.save_dir):
    tf.gfile.MakeDirs(FLAGS.save_dir)

  with tf.Graph().as_default():
    datasets = dataset.get_datasets()
    dataset_content = datasets[FLAGS.dataset](
        FLAGS.dataset, FLAGS.dataset_split, 4, 128 * 1024)
    batched = dataset_content.batch(64)
    batch_op = batched.make_one_shot_iterator().get_next()

    with tf.Session() as session:
      data = session.run(batch_op)[0]

  if not FLAGS.grid:
    for i in range(data.shape[0]):
      save_image(data[i], FLAGS.dataset + "_%d.png" % i)
  else:
    grid_im = []
    for i in range(8):
      im_row = []
      for j in range(8):
        im_row.append(data[i * 8 + j])
      grid_im.append(np.concatenate(im_row, axis=1))
    grid_im = np.concatenate(grid_im, axis=0)
    save_image(grid_im, FLAGS.dataset + "_grid.png")

if __name__ == "__main__":
  tf.app.run(main)
