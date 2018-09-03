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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan.src import fid_score
import tensorflow as tf


class FidScoreTest(tf.test.TestCase):

  def _create_fake_inception_graph(self):
    """Creates a graph with that mocks inception.

    It takes the input, multiplies it through a matrix full of 0.001 values
    and returns as logits. It makes sure to match the tensor names of
    the real inception model.

    Returns:
      tf.Graph object with a simple mock inception inside.
    """
    fake_inception = tf.Graph()
    with fake_inception.as_default():
      graph_input = tf.placeholder(
          tf.float32, shape=[None, 299, 299, 3], name="Mul")
      matrix = tf.ones(shape=[299 * 299 * 3, 10]) * 0.001
      output = tf.matmul(tf.layers.flatten(graph_input), matrix)
      output = tf.identity(output, name="pool_3")
      output = tf.identity(output, name="logits")
    return fake_inception

  def test_fid_function(self):
    with tf.Graph().as_default():
      real_data = tf.zeros((10, 4, 4, 3))
      gen_data = tf.ones((10, 4, 4, 3))
      inception_graph = self._create_fake_inception_graph()

      eval_fn = fid_score.get_fid_function(real_data, gen_data, 18, 19, "0_255",
                                           inception_graph.as_graph_def())
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = eval_fn(sess)
        self.assertNear(result, 43.8, 0.1)


if __name__ == "__main__":
  tf.test.main()
