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

"""Discriminator accuracy.

Computes the discrimionator's accuracy on (a subset) of the training dataset,
test dataset, and a generated data set. The score is averaged over several
multiple generated data sets and subsets of the training data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from compare_gan import datasets
from compare_gan import eval_utils
from compare_gan.metrics import eval_task

import numpy as np


class AccuracyTask(eval_task.EvalTask):
  """Evaluation Task for computing and reporting accuracy."""

  def metric_list(self):
    return frozenset([
        "train_accuracy", "test_accuracy", "fake_accuracy", "train_d_loss",
        "test_d_loss"
    ])

  def run_in_session(self, options, sess, gan, real_images):
    del options
    return compute_accuracy_loss(sess, gan, real_images)


def compute_accuracy_loss(sess,
                          gan,
                          test_images,
                          max_train_examples=50000,
                          num_repeat=5):
  """Compute discriminator's accuracy and loss on a given dataset.

  Args:
    sess: Tf.Session object.
    gan: Any AbstractGAN instance.
    test_images: numpy array with test images.
    max_train_examples: How many "train" examples to get from the dataset.
                        In each round, some of them will be randomly selected
                        to evaluate train set accuracy.
    num_repeat: How many times to repreat the computation.
                The mean of all the results is reported.
  Returns:
    Dict[Text, float] with all the computed scores.

  Raises:
    ValueError: If the number of test_images is greater than the number of
                training images returned by the dataset.
  """
  logging.info("Evaluating training and test accuracy...")
  train_images = eval_utils.get_real_images(
      dataset=datasets.get_dataset(),
      num_examples=max_train_examples,
      split="train",
      failure_on_insufficient_examples=False)
  if train_images.shape[0] < test_images.shape[0]:
    raise ValueError("num_train %d must be larger than num_test %d." %
                     (train_images.shape[0], test_images.shape[0]))

  num_batches = int(np.floor(test_images.shape[0] / gan.batch_size))
  if num_batches * gan.batch_size < test_images.shape[0]:
    logging.error("Ignoring the last batch with %d samples / %d epoch size.",
                  test_images.shape[0] - num_batches * gan.batch_size,
                  gan.batch_size)

  ret = {
      "train_accuracy": [],
      "test_accuracy": [],
      "fake_accuracy": [],
      "train_d_loss": [],
      "test_d_loss": []
  }

  for _ in range(num_repeat):
    idx = np.random.choice(train_images.shape[0], test_images.shape[0])
    bs = gan.batch_size
    train_subset = [train_images[i] for i in idx]
    train_predictions, test_predictions, fake_predictions = [], [], []
    train_d_losses, test_d_losses = [], []

    for i in range(num_batches):
      z_sample = gan.z_generator(gan.batch_size, gan.z_dim)
      start_idx = i * bs
      end_idx = start_idx + bs
      test_batch = test_images[start_idx : end_idx]
      train_batch = train_subset[start_idx : end_idx]

      test_prediction, test_d_loss, fake_images = sess.run(
          [gan.discriminator_output, gan.d_loss, gan.fake_images],
          feed_dict={
              gan.inputs: test_batch, gan.z: z_sample
          })
      train_prediction, train_d_loss = sess.run(
          [gan.discriminator_output, gan.d_loss],
          feed_dict={
              gan.inputs: train_batch,
              gan.z: z_sample
          })
      fake_prediction = sess.run(
          gan.discriminator_output,
          feed_dict={gan.inputs: fake_images})[0]

      train_predictions.append(train_prediction[0])
      test_predictions.append(test_prediction[0])
      fake_predictions.append(fake_prediction)
      train_d_losses.append(train_d_loss)
      test_d_losses.append(test_d_loss)

    train_predictions = [x >= 0.5 for x in train_predictions]
    test_predictions = [x >= 0.5 for x in test_predictions]
    fake_predictions = [x < 0.5 for x in fake_predictions]

    ret["train_accuracy"].append(np.array(train_predictions).mean())
    ret["test_accuracy"].append(np.array(test_predictions).mean())
    ret["fake_accuracy"].append(np.array(fake_predictions).mean())
    ret["train_d_loss"].append(np.mean(train_d_losses))
    ret["test_d_loss"].append(np.mean(test_d_losses))

  for key in ret:
    ret[key] = np.mean(ret[key])

  return ret
