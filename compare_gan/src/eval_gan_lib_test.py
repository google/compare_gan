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

import csv
import os
import os.path

from compare_gan.src import eval_gan_lib
from compare_gan.src import fid_score as fid_score_lib
from compare_gan.src import gan_lib
from compare_gan.src.gans import abstract_gan
from compare_gan.src.gans import consts
import mock
import numpy as np
import tensorflow as tf


class EvalGanLibTest(tf.test.TestCase):

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

  def _create_checkpoint(self, variable_name, checkpoint_path):
    """Creates a checkpoint with a single variable stored inside.

    There must be at least one variable in the checkpoint for tf.Saver
    to work correctly.

    Args:
      variable_name: string, name of the tf variable to save in the checkpoint.
      checkpoint_path: string, path to where checkpoint will be written.
    """
    # Create a checkpoint with a single variable.
    with tf.Graph().as_default():
      tf.get_variable(variable_name, shape=[1])
      saver = tf.train.Saver()
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_path)

  def _get_scores(self, workdir):
    """Returns all the scores from the csv file in a list."""
    with tf.gfile.FastGFile(os.path.join(workdir, "scores.csv")) as csvfile:
      return list(csv.DictReader(csvfile))

  def test_end2end_checkpoint(self):
    """Takes real GAN (trained for 1 step) and evaluate it."""
    workdir = os.path.join(tf.test.get_temp_dir(), self.id())
    tf.logging.info("Workdir: %s" % workdir)
    options = {
        "gan_type": "GAN",
        "dataset": "fake",
        "training_steps": 1,
        "save_checkpoint_steps": 10,
        "learning_rate": 0.001,
        "discriminator_normalization": consts.NO_NORMALIZATION,
        "eval_test_samples": 50,
    }
    gan_lib.run_with_options(options, workdir)
    fake_inception = self._create_fake_inception_graph()

    eval_gan_lib.RunTaskEval(
        options, workdir, inception_graph=fake_inception.as_graph_def())

    rows = self._get_scores(workdir)

    self.assertEquals(1, len(rows))
    # The fid score should exist (and be quite large).
    self.assertGreater(rows[0]["fid_score"], 100.0)

  _FAKE_FID_SCORE = 18.3

  def test_mock_gan_and_mock_fid(self):
    """Mocks out the GAN and eval tasks and check that eval still works."""
    workdir = os.path.join(tf.test.get_temp_dir(), self.id())
    tf.logging.info("Workdir: %s" % workdir)
    options = {
        "gan_type": "GAN", "dataset": "fake", "eval_test_samples": 50,
        "gilbo_max_train_cycles": 5, "gilbo_train_steps_per_cycle": 10,
        "gilbo_eval_steps": 10, "compute_gilbo": True,
    }
    checkpoint_path = os.path.join(workdir, "model")
    self._create_checkpoint("foo", checkpoint_path)

    with mock.patch.object(gan_lib, "create_gan", autospec=True) as mock_cls:
      with mock.patch.object(
          fid_score_lib, "get_fid_function", autospec=True) as fid_mock:
        fid_mock.return_value.return_value = self._FAKE_FID_SCORE

        mock_gan = mock_cls.return_value
        mock_gan.batch_size = 16

        def create_mock_gan(is_training):
          """Creates a minimal graph that has all the required GAN nodes.

          It also has a single (unused) variable inside, to make sure that
          tf.Saver is happy.

          Args:
            is_training: unused, but required by mock.
          """
          del is_training
          z = tf.placeholder(tf.float32)
          mock_gan.z = z
          tf.get_variable("foo", shape=[1])
          fake_images = tf.ones([16, 64, 64, 1])
          mock_gan.fake_images = fake_images
          mock_gan.z_dim = 64

        mock_gan.build_model.side_effect = create_mock_gan
        fake_inception = self._create_fake_inception_graph()
        inception_graph = fake_inception.as_graph_def()

        tasks_to_run = [
            eval_gan_lib.InceptionScoreTask(inception_graph),
            eval_gan_lib.FIDScoreTask(inception_graph),
            eval_gan_lib.MultiscaleSSIMTask(),
            eval_gan_lib.ComputeAccuracyTask(),
            eval_gan_lib.GILBOTask(workdir, workdir, options["dataset"]),
        ]

        result_dict = eval_gan_lib.RunCheckpointEval(checkpoint_path, workdir,
                                                     options, tasks_to_run)
        self.assertEquals(result_dict["fid_score"], self._FAKE_FID_SCORE)

        print(result_dict)
        self.assertNear(result_dict["gilbo"], 0.0, 5.0,
                        "GILBO should be pretty close to 0!")

  def test_mock_gan_and_nan(self):
    """Tests the case, where GAN starts returning NaNs."""
    workdir = os.path.join(tf.test.get_temp_dir(), self.id())
    tf.logging.info("Workdir: %s" % workdir)
    options = {"gan_type": "GAN", "dataset": "fake", "eval_test_samples": 50}
    checkpoint_path = os.path.join(workdir, "model")
    self._create_checkpoint("foo", checkpoint_path)

    with mock.patch.object(gan_lib, "create_gan", autospec=True) as mock_cls:
      mock_gan = mock_cls.return_value
      mock_gan.batch_size = 16

      def create_mock_gan(is_training):
        """Creates a minimal graph that has all the required GAN nodes.

        It will return a NaN values in the output.

        Args:
          is_training: unused but required by mock.
        """
        del is_training
        z = tf.placeholder(tf.float32)
        mock_gan.z = z
        tf.get_variable("foo", shape=[1])
        fake_images = tf.ones([16, 64, 64, 1]) * float("NaN")
        mock_gan.fake_images = fake_images

      mock_gan.build_model.side_effect = create_mock_gan
      with self.assertRaises(eval_gan_lib.NanFoundError):
        eval_gan_lib.RunCheckpointEval(
            checkpoint_path, workdir, options, tasks_to_run=[])

  def test_accuracy_loss(self):
    """Evaluates accuracy loss metric on mock graph."""
    options = {
        "gan_type": "GAN",
        "dataset": "fake",
    }
    mock_gan = mock.create_autospec(abstract_gan.AbstractGAN)
    mock_gan.batch_size = 100
    with tf.Graph().as_default():
      with tf.Session() as sess:
        test_images = np.random.random(size=(100, 64, 64, 1))
        mock_gan.z = tf.placeholder(tf.float32)
        mock_gan.inputs = tf.placeholder(tf.float32)
        mock_gan.fake_images = tf.ones([100, 64, 64, 1])
        mock_gan.discriminator_output = [tf.constant(0.3)]
        mock_gan.d_loss = tf.constant(1.1)
        mock_gan.z_dim = 15
        mock_gan.z_generator.return_value = [1.0, 2.0, 3.0]
        result_dict = eval_gan_lib.ComputeAccuracyLoss(
            options,
            sess,
            mock_gan,
            test_images,
            max_train_examples=100,
            num_repeat=1)
        # Model always returns 0.3 (so it thinks that image is rather fake)
        # On test/train, it means 0% accuracy as the threshold is 0.5.
        # On fake, it is going to have 100% accuracy,
        self.assertNear(result_dict["train_accuracy"], 0.0, 0.01)
        self.assertNear(result_dict["test_accuracy"], 0.0, 0.01)
        self.assertNear(result_dict["fake_accuracy"], 1.0, 0.01)
        self.assertNear(result_dict["train_d_loss"], 1.1, 0.01)
        self.assertNear(result_dict["test_d_loss"], 1.1, 0.01)
        # Make sure that the keys are matchhing the MetricsList from the task
        self.assertSetEqual(eval_gan_lib.ComputeAccuracyTask().MetricsList(),
                            set(result_dict.keys()))

  def test_condition_number_for_mock(self):
    """Tests that condition number task runs end to end without crashing."""
    workdir = os.path.join(tf.test.get_temp_dir(), self.id())
    tf.logging.info("Workdir: %s" % workdir)
    options = {
        "gan_type": "GAN",
        "dataset": "fake",
        "eval_test_samples": 50,
        "compute_generator_condition_number": True
    }
    checkpoint_path = os.path.join(workdir, "model")
    self._create_checkpoint("foo", checkpoint_path)

    with mock.patch.object(gan_lib, "create_gan", autospec=True) as mock_cls:
      mock_gan = mock_cls.return_value
      mock_gan.batch_size = 16
      mock_gan.z_dim = 2

      def z_generator(batch_size, z_dim):
        return np.random.uniform(
            -1, 1, size=(batch_size, z_dim)).astype(np.float32)

      mock_gan.z_generator = z_generator

      def create_mock_gan(is_training):
        """Creates a minimal graph that has all the required GAN nodes.

        It also has a single (unused) variable inside, to make sure that
        tf.Saver is happy.

        Args:
          is_training: unused, but required by mock.
        """
        del is_training
        z = tf.placeholder(
            tf.float32, shape=(mock_gan.batch_size, mock_gan.z_dim))
        mock_gan.z = z
        # Trivial function from z to fake images to compute Jacobian of.
        fake_images = tf.tile(
            tf.reshape(mock_gan.z, [16, 2, 1, 1]), [1, 32, 64, 1])
        mock_gan.fake_images = fake_images
        tf.get_variable("foo", shape=[1])

      mock_gan.build_model.side_effect = create_mock_gan

      tasks_to_run = [
          eval_gan_lib.GeneratorConditionNumberTask(),
      ]

      result_dict = eval_gan_lib.RunCheckpointEval(checkpoint_path, workdir,
                                                   options, tasks_to_run)
      self.assertEquals(result_dict["log_condition_number_count"], 16)
      self.assertEquals(result_dict["log_condition_number_mean"], 0)
      self.assertEquals(result_dict["log_condition_number_std"], 0)

  def test_csv_writing(self):
    """Verifies that results are correctly written to final CSV file."""
    workdir = os.path.join(tf.test.get_temp_dir(), self.id())
    checkpoint_path = os.path.join(workdir, "checkpoint/")
    tf.gfile.MakeDirs(checkpoint_path)

    options = {
        "gan_type": "GAN",
        "dataset": "fake",
        "discriminator_normalization": consts.NO_NORMALIZATION,
        "learning_rate": 0.001,
    }
    # Create 10 checkpoints.
    with tf.Graph().as_default():
      tf.get_variable("foo", shape=[1])
      saver = tf.train.Saver(max_to_keep=1000)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for x in range(10):
          saver.save(sess, checkpoint_path, global_step=x)

    with mock.patch.object(
        eval_gan_lib, "RunCheckpointEval", autospec=True) as mock_cls:
      result_dict = {"inception_score": 12.0, "train_d_loss": 1.3}
      mock_cls.return_value = result_dict
      eval_gan_lib.RunTaskEval(options, workdir, inception_graph=None)
    rows = self._get_scores(workdir)
    self.assertEquals(10, len(rows))
    self.assertNear(float(rows[0]["inception_score"]), 12.0, 0.01)
    self.assertNear(float(rows[1]["train_d_loss"]), 1.3, 0.01)
    self.assertNear(float(rows[1]["test_accuracy"]), -1.0, 0.01)

  def test_csv_append(self):
    workdir = os.path.join(tf.test.get_temp_dir(), self.id())
    checkpoint_path = os.path.join(workdir, "checkpoint/")
    tf.gfile.MakeDirs(checkpoint_path)

    options = {
        "gan_type": "GAN",
        "dataset": "fake",
        "discriminator_normalization": consts.NO_NORMALIZATION,
        "learning_rate": 0.001,
    }

    # Start by creating first 2 checkpoints.
    with tf.Graph().as_default():
      tf.get_variable("foo", shape=[1])
      saver = tf.train.Saver(max_to_keep=1000)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for x in range(2):
          saver.save(sess, checkpoint_path, global_step=x)

        with mock.patch.object(
            eval_gan_lib, "RunCheckpointEval", autospec=True) as mock_cls:
          mock_cls.return_value = {"inception_score": 12.0, "train_d_loss": 1.3}
          eval_gan_lib.RunTaskEval(options, workdir, inception_graph=None)

          rows = self._get_scores(workdir)
          self.assertEquals(2, len(rows))
          self.assertNear(float(rows[0]["inception_score"]), 12.0, 0.01)
          self.assertNear(float(rows[1]["train_d_loss"]), 1.3, 0.01)
          self.assertNear(float(rows[1]["test_accuracy"]), -1.0, 0.01)

          # Now create 2 more checkpoints.
          for x in range(3, 5):
            saver.save(sess, checkpoint_path, global_step=x)
          mock_cls.return_value = {"inception_score": 14.0, "train_d_loss": 1.5}
          eval_gan_lib.RunTaskEval(options, workdir, inception_graph=None)
          rows = self._get_scores(workdir)
          self.assertEquals(4, len(rows))
          # old scores should stay intact.
          self.assertNear(float(rows[0]["inception_score"]), 12.0, 0.01)
          self.assertNear(float(rows[1]["train_d_loss"]), 1.3, 0.01)
          self.assertNear(float(rows[1]["test_accuracy"]), -1.0, 0.01)
          # New entries should have new values.
          self.assertNear(float(rows[2]["inception_score"]), 14.0, 0.01)
          self.assertNear(float(rows[3]["train_d_loss"]), 1.5, 0.01)

          self.assertNotIn("new_metric", rows[0])

          # Now assume that metric names have changed.
          with mock.patch.object(
              eval_gan_lib, "MultiscaleSSIMTask", autospec=True) as mock_task:
            mock_task.return_value.MetricsList.return_value = [
                "ms_ssim", "new_metric"
            ]
            # Now create 2 more checkpoints.
            for x in range(5, 7):
              saver.save(sess, checkpoint_path, global_step=x)
            mock_cls.return_value = {
                "inception_score": 16.0,
                "train_d_loss": 1.7,
                "new_metric": 20.0
            }
            eval_gan_lib.RunTaskEval(options, workdir, inception_graph=None)
            rows = self._get_scores(workdir)
            self.assertEquals(6, len(rows))

            # As CSV header has changed, all the results should have been
            # recomputed.
            for x in range(6):
              self.assertNear(float(rows[x]["inception_score"]), 16.0, 0.01)
              self.assertNear(float(rows[x]["new_metric"]), 20.0, 0.01)
              self.assertNear(float(rows[x]["test_accuracy"]), -1.0, 0.01)
              self.assertNear(float(rows[x]["train_d_loss"]), 1.7, 0.01)

  def test_save_final(self):
    workdir = os.path.join(tf.test.get_temp_dir(), self.id())
    tf.gfile.MakeDirs(workdir)
    scores_path = os.path.join(workdir, "scores")
    value_path = os.path.join(workdir, "value")

    with tf.gfile.FastGFile(scores_path, "w") as csvfile:
      writer = csv.DictWriter(csvfile, ["fid_score", "other_score"])
      writer.writeheader()
      writer.writerow({"fid_score": 13.0, "other_score": 20.0})
      writer.writerow({"fid_score": 17.0, "other_score": 10.0})

    eval_gan_lib.SaveFinalEvaluationScore(scores_path, "other_score",
                                          value_path)

    with tf.gfile.FastGFile(value_path) as f:
      self.assertEquals(f.read(), "10.0")


class FractalDimensionTest(tf.test.TestCase):

  def test_straight_line(self):
    """The fractal dimension of a 1D line mustlie near 1.0 ."""

    self.assertAllClose(
        eval_gan_lib.ComputeFractalDimension(
            np.random.uniform(size=(10000, 1))),
        1.0,
        atol=0.05)

  def test_square(self):
    """The fractal dimension of a 2D square must lie near 2.0 ."""

    self.assertAllClose(
        eval_gan_lib.ComputeFractalDimension(
            np.random.uniform(size=(10000, 2))),
        2.0,
        atol=0.1)


if __name__ == "__main__":
  tf.test.main()
