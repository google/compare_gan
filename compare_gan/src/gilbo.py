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

"""Implements GILBO score functions.

See https://arxiv.org/abs/1802.04874 for details.

See eval_gan_lib.py for working code that calls into this code.

It should be sufficient to call in the following manner:
```python

dataset = datasets.load_mnist(...)
gan = gans.GAN(...)

sess = tf.Session(...)

# Train GAN...

outdir = '/desired/output/path'
checkpoint_path = '/path/to/gan/checkpoints'
options = dict(
    'gilbo_learning_rate': 4e-4,
    ... # Other GILBO-specific options visible in this file.
)

gilbo, = TrainGILBO(
    gan, sess, outdir, checkpoint_path, dataset, options)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os

from compare_gan.src import gan_lib
from compare_gan.src.gans.VAE import VAE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

import numpy as np
from pstar import plist
import scipy.misc
import tensorflow as tf

layers = tf.layers
ds = tf.contrib.distributions


def _Regressor(x, z_dim=64, gan_type='GAN'):
  """Make the GILBO regressor, which is based off of the GAN discriminator."""
  assert gan_type in ['GAN', 'VAE']

  net = tf.cast(x, tf.float32)
  net = layers.conv2d(net, 64, 4, 2, activation=tf.nn.leaky_relu)
  net = layers.conv2d(net, 128, 4, 2, activation=tf.nn.leaky_relu)
  net = layers.flatten(net)
  net = layers.dense(net, 1024, activation=tf.nn.leaky_relu)
  net = layers.dense(net, 2 * z_dim)

  if gan_type == 'VAE':
    encoding = ds.Normal(
        loc=net[..., :z_dim],
        scale=tf.nn.softplus(net[..., z_dim:]))
    dist = ds.Independent(encoding, 1)
    return dist
  else:
    # a is the alpha parameter and b is the beta parameter of the Beta
    # distribution.
    a, b = net[..., :z_dim], net[..., z_dim:2 * z_dim]

    a = 1 + tf.nn.softplus(a - 5)
    b = 1 + tf.nn.softplus(b - 5)

    dist = ds.Independent(ds.Beta(a, b), 1)
    bijector = ds.bijectors.Affine(-1.0, 2.0)
    tdist = ds.TransformedDistribution(distribution=dist, bijector=bijector)
    return tdist


def _SaveImage(img, filename):
  # If img is [H W] or [H W 1], stack into [H W 3] for scipy's api.
  if len(img.shape) == 2 or img.shape[-1] == 1:
    img = np.stack((img.squeeze(),) * 3, -1)
  with tf.gfile.Open(filename, 'w') as f:
    scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(f)


def _SaveZHistograms(gan, z_sample, z_pred_dist, gan_type, outdir, step):
  """Save a histogram for each z dimension as an png in outdir."""
  fig, axs = plt.subplots(8, 8, figsize=(15, 10))

  pk = 0
  bins = np.linspace(-1, 1, 70)

  samp = z_sample.eval()
  z_pred_samp = z_pred_dist.sample(10000).eval({gan.z: samp})

  try:
    for j in range(64):
      if gan_type == 'VAE':
        bins = np.linspace(np.nan_to_num(z_pred_samp[:, pk, j]).min(),
                           np.nan_to_num(z_pred_samp[:, pk, j]).max(), 70)
      axs.flat[j].hist(z_pred_samp[:, pk, j], bins, histtype='stepfilled',
                       normed=True)
      axs.flat[j].vlines(samp[pk, j], 0, 1.0, linestyle='dashed')

    plt.tight_layout()
    filename = os.path.join(outdir, 'z_hist_%03d.png' % step)
    tf.logging.info('Saving z histogram: %s' % filename)
    with tf.gfile.Open(filename, 'w') as f:
      fig.savefig(f, dpi='figure')
  except Exception as e:  # pylint: disable=broad-except
    tf.logging.info('Caught %r while rendering chart. Ignoring.\n%s\n'
                    % (type(e), str(e)))


def _SaveGILBO(saver, sess, learning_rate, gilbo_step, step, lr, outdir):
  """Save GILBO model checkpoints, including the current step and lr.

  Args:
    saver: tf.train.Saver.
    sess: tf.Session.
    learning_rate: tf.Variable for the learning rate.
    gilbo_step: tf.Variable for the current training step.
    step: integer for the current step, to be saved in the checkpoint.
    lr: float for the current learning rate, to be saved in the checkpoint.
    outdir: output directory.
  """
  # Save the current learning rate and gilbo training step with the checkpoint.
  learning_rate.assign(lr).eval()
  gilbo_step.assign(step).eval()
  filename = os.path.join(outdir, 'gilbo_model')
  saver.save(sess, filename, global_step=step)


def TrainGILBO(gan, sess, outdir, checkpoint_path, dataset, options):
  """Build and train GILBO model.

  Args:
    gan: GAN object.
    sess: tf.Session.
    outdir: Output directory. A pickle file will be written there.
    checkpoint_path: Path where gan's checkpoints are written. Only used to
                     ensure that GILBO files are written to a unique
                     subdirectory of outdir.
    dataset: Name of dataset used to train the GAN.
    options: Options dictionary.

  Returns:
    mean_eval_info: Mean GILBO computed over a large number of images generated
                    by the trained GAN or VAE.
    mean_train_consistency: Mean consistency of the trained GILBO model with
                            data from the training set.
    mean_eval_consistency: Same consistency measure for the trained model with
                           data from the validation set.
    mean_self_consistency: Same consistency measure for the trained model with
                           data generated by the trained model itself.
    See the GILBO paper for an explanation of these metrics.

  Raises:
    ValueError: If the GAN has uninitialized variables.
  """
  uninitialized = sess.run(tf.report_uninitialized_variables())
  if uninitialized:
    raise ValueError('Model has uninitialized variables!\n%r' % uninitialized)

  outdir = os.path.join(outdir, checkpoint_path.replace('/', '_'))
  if isinstance(gan, VAE):
    gan_type = 'VAE'
  else:
    gan_type = 'GAN'

  tf.gfile.MakeDirs(outdir)
  with tf.variable_scope('gilbo'):
    ones = tf.ones((gan.batch_size, gan.z_dim))
    # Get a distribution for the prior, depending on whether the model is a VAE
    # or a GAN.
    if gan_type == 'VAE':
      z_dist = ds.Independent(ds.Normal(0.0 * ones, ones), 1)
    else:
      z_dist = ds.Independent(ds.Uniform(-ones, ones), 1)

    z_sample = z_dist.sample()

    epsneg = np.finfo('float32').epsneg

    if gan_type == 'VAE':
      ganz_clip = gan.z
    else:
      # Clip samples from the GAN uniform prior because the Beta distribution
      # doesn't include the top endpoint and has issues with the bottom endpoint
      ganz_clip = tf.clip_by_value(gan.z, -(1 - epsneg), 1 - epsneg)

    # Get generated images from the model.
    fake_images = gan.fake_images

    # Build the regressor distribution that encodes images back to predicted
    # samples from the prior.
    with tf.variable_scope('regressor'):
      z_pred_dist = _Regressor(fake_images, gan.z_dim, gan_type)

    # Capture the parameters of the distributions for later analysis.
    if gan_type == 'VAE':
      dist_p1 = z_pred_dist.distribution.loc
      dist_p2 = z_pred_dist.distribution.scale
    else:
      dist_p1 = z_pred_dist.distribution.distribution.concentration0
      dist_p2 = z_pred_dist.distribution.distribution.concentration1

    # info and avg_info compute the GILBO.
    info = z_pred_dist.log_prob(ganz_clip) - z_dist.log_prob(ganz_clip)
    avg_info = tf.reduce_mean(info)

    # Set up training of the GILBO model.
    lr = options.get('gilbo_learning_rate', 4e-4)
    learning_rate = tf.get_variable(
        'learning_rate', initializer=lr, trainable=False)
    gilbo_step = tf.get_variable('gilbo_step', dtype=tf.int32, initializer=0,
                                 trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate)

    regressor_vars = tf.contrib.framework.get_variables('gilbo/regressor')
    train_op = opt.minimize(-info, var_list=regressor_vars)

  # Initialize the variables we just created.
  uninitialized = plist(tf.report_uninitialized_variables().eval())
  uninitialized_vars = uninitialized.apply(
      tf.contrib.framework.get_variables_by_name)._[0]
  tf.variables_initializer(uninitialized_vars).run()

  saver = tf.train.Saver(uninitialized_vars, max_to_keep=1)
  try:
    checkpoint_path = tf.train.latest_checkpoint(outdir)
    saver.restore(sess, checkpoint_path)
  except ValueError:
    # Failing to restore just indicates that we don't have a valid checkpoint,
    # so we will just start training a fresh GILBO model.
    pass

  _TrainGILBO(sess, gan, saver, learning_rate, gilbo_step, z_sample, avg_info,
              z_pred_dist, train_op, gan_type, outdir, options)

  mean_eval_info = _EvalGILBO(
      sess, gan, z_sample, avg_info, dist_p1, dist_p2, fake_images, outdir,
      options)

  # Collect encoded distributions on the training and eval set in order to do
  # kl-nearest-neighbors on generated samples and measure consistency.
  train_images = gan_lib.load_dataset(
      dataset, split_name='train'
  ).apply(
      tf.contrib.data.batch_and_drop_remainder(gan.batch_size)
  ).make_one_shot_iterator().get_next()[0]
  train_images = tf.reshape(train_images, fake_images.shape)

  eval_images = gan_lib.load_dataset(
      dataset, split_name='test'
  ).apply(
      tf.contrib.data.batch_and_drop_remainder(gan.batch_size)
  ).make_one_shot_iterator().get_next()[0]
  eval_images = tf.reshape(eval_images, fake_images.shape)

  mean_train_consistency = _RunGILBOConsistency(
      train_images, 'train', extract_input_images=0,
      save_consistency_images=20, num_batches=5, **locals())
  mean_eval_consistency = _RunGILBOConsistency(
      eval_images, 'eval', extract_input_images=0,
      save_consistency_images=20, num_batches=5, **locals())
  mean_self_consistency = _RunGILBOConsistency(
      fake_images, 'self', extract_input_images=20,
      save_consistency_images=20, num_batches=5, **locals())

  return (mean_eval_info,
          mean_train_consistency,
          mean_eval_consistency,
          mean_self_consistency)


def _TrainGILBO(sess, gan, saver, learning_rate, gilbo_step, z_sample, avg_info,
                z_pred_dist, train_op, gan_type, outdir, options):
  """Run the training process."""
  lr_scale = options.get('gilbo_lr_scale', 0.5)
  min_lr = options.get('gilbo_min_lr', 1e-8)
  min_ai_step_scale = options.get('gilbo_min_ai_step_scale', 0.75)
  min_ai_step_value = options.get('gilbo_min_ai_step_value', 0.5)
  max_train_cycles = options.get('gilbo_max_train_cycles', 50)
  train_steps_per_cycle = options.get('gilbo_train_steps_per_cycle', 10000)

  ais = [0.0]  # average gilbos (i is for info)
  min_ai = -2.0
  lr, i = sess.run([learning_rate, gilbo_step])
  for i in range(i, max_train_cycles):
    if lr < min_lr:
      break

    _SaveGILBO(saver, sess, learning_rate, gilbo_step, i, lr, outdir)
    ai = 0.0
    for j in range(train_steps_per_cycle):
      if j % (train_steps_per_cycle // 10) == 0:
        tf.logging.info('step:%d, gilbo:%.3f' % (j, ai))
      samp = sess.run(z_sample)

      _, z_info = sess.run(
          [train_op, avg_info],
          feed_dict={gan.z: samp, learning_rate: lr})
      ai += (z_info - ai) / (j + 1)
    tf.logging.info('cycle:%d gilbo:%.3f min next gilbo:%.3f learning rate:%.3f'
                    % (i, ai, min_ai, lr))

    if ai < min_ai:
      lr *= lr_scale
    if lr < min_lr:
      break
    if np.isnan(ai):
      tf.logging.info('NaN GILBO at cycle %d, stopping training early.' % i)
      break

    ais.append(ai)

    # min_ai is the minimum next GILBO for the training algorithm to consider
    # that progress is being made. GILBO is a lower bound that we are maximizing
    # so we want it to increase during each training cycle.
    min_ai = max(min_ai,
                 ai + max(0.0,
                          min(min_ai_step_value,
                              (ai - ais[-2]) * min_ai_step_scale)
                         )
                )

  _SaveGILBO(saver, sess, learning_rate, gilbo_step, i, lr, outdir)

  _SaveZHistograms(gan, z_sample, z_pred_dist, gan_type, outdir, i)


def _EvalGILBO(sess, gan, z_sample, avg_info, dist_p1, dist_p2, fake_images,
               outdir, options):
  """Evaluate GILBO on new data from the generative model.

  Args:
    sess: tf.Session.
    gan: GAN object.
    z_sample: Tensor sampling from the prior.
    avg_info: Tensor that computes the per-batch GILBO.
    dist_p1: Tensor for the first parameter of the distribution
             (e.g., concentration1 for a Beta distribution).
    dist_p2: Tensor for the second parameter of the distribution
             (e.g., concentration2 for a Beta distribution).
    fake_images: Tensor of images sampled from the GAN.
    outdir: Output directory. A pickle file will be written there.
    options: Options dictionary.

  Returns:
    The mean GILBO on the evaluation set. Also writes a pickle file saving
    distribution parameters and generated images for later analysis.
  """
  eval_steps = options.get('gilbo_eval_steps', 10000)
  z_infos = np.zeros(eval_steps, np.float32)
  z_dist_p1s, z_dist_p2s, z_fake_images = [], [], []
  mean_eval_info = 0
  for i in range(eval_steps):
    samp = sess.run(z_sample)
    if i * gan.batch_size < 1000:
      # Save the first 1000 distribution parameters and generated images for
      # separate data processing.
      z_infos[i], z_dist_p1, z_dist_p2, images = sess.run(
          [avg_info, dist_p1, dist_p2, fake_images], feed_dict={gan.z: samp})
      z_dist_p1s.append(z_dist_p1)
      z_dist_p2s.append(z_dist_p2)
      z_fake_images.append(images)
    else:
      z_infos[i] = sess.run(avg_info, feed_dict={gan.z: samp})

    if i % (eval_steps // 10) == 0:
      tf.logging.info('eval step:%d gilbo:%3.1f' % (i, z_infos[i]))

  if eval_steps:
    mean_eval_info = np.mean(np.nan_to_num(z_infos))
    eval_dists = dict(
        dist_p1=np.array(z_dist_p1s).reshape([-1, 64]),
        dist_p2=np.array(z_dist_p2s).reshape([-1, 64]),
        images=np.array(z_fake_images).reshape(
            [-1] + list(z_fake_images[0].shape[1:])))
    with tf.gfile.Open(os.path.join(outdir, 'eval_dists.p'), 'w') as f:
      pickle.dump(eval_dists, f)
    tf.logging.info('eval gilbo:%3.1f' % mean_eval_info)

  return mean_eval_info


def _RunGILBOConsistency(input_images, mode, dist_p1, dist_p2, z_pred_dist,
                         z_sample, gan, sess, outdir, gan_type, dataset,
                         extract_input_images=0,
                         save_consistency_images=0,
                         num_batches=3000,
                         **unused_kw):
  """Measure consistency of the gilbo estimator with the GAN or VAE.

  Arguments without documentation are variables from the calling function needed
  here. Pass them with **locals().

  Args:
    input_images: Tensor. Dataset images, or images generated by the GAN or VAE.
    mode: 'train', 'eval', or 'self'. Which consistency measure to compute.
    dist_p1:
    dist_p2:
    z_pred_dist:
    z_sample:
    gan:
    sess:
    outdir:
    gan_type:
    dataset:
    extract_input_images: Number of batches to extract, -1 for all. Default: 0.
    save_consistency_images: Num batches to save, -1 for all. Default: 0.
    num_batches: Number of batches to run. Default: 3000.
    **unused_kw: Unused extra keyword args.

  Returns:
    Symmetric consistency KL. Additionally saves distribution parameters as a
    pickle as well as any requested images as pngs to outdir.
  """
  with tf.variable_scope('gilbo'):
    with tf.variable_scope('regressor', reuse=True):
      z_pred_dist_train = _Regressor(input_images, gan.z_dim, gan_type)

      z_sample_train = z_pred_dist_train.sample()

    dist_p1_ph = tf.placeholder(tf.float32, dist_p1.shape)
    dist_p2_ph = tf.placeholder(tf.float32, dist_p2.shape)
    consist_dist_p1_ph = tf.placeholder(tf.float32, dist_p1.shape)
    consist_dist_p2_ph = tf.placeholder(tf.float32, dist_p2.shape)

    if gan_type == 'VAE':
      dist_p1 = z_pred_dist_train.distribution.loc
      dist_p2 = z_pred_dist_train.distribution.scale

      consist_z_dist_p1 = z_pred_dist.distribution.loc
      consist_z_dist_p2 = z_pred_dist.distribution.scale

      base_dist = ds.Normal
    else:
      dist_p1 = z_pred_dist_train.distribution.distribution.concentration0
      dist_p2 = z_pred_dist_train.distribution.distribution.concentration1

      consist_z_dist_p1 = z_pred_dist.distribution.distribution.concentration0
      consist_z_dist_p2 = z_pred_dist.distribution.distribution.concentration1

      base_dist = ds.Beta

    kl_dist_p = ds.Independent(
        base_dist(dist_p1_ph, dist_p2_ph), 1)
    kl_dist_q = ds.Independent(
        base_dist(consist_dist_p1_ph, consist_dist_p2_ph), 1)

    consistency_kl = kl_dist_p.kl_divergence(kl_dist_q)
    consistency_rkl = kl_dist_q.kl_divergence(kl_dist_p)

  z_dist_p1s, z_dist_p2s = [], []
  consist_z_dist_p1s, consist_z_dist_p2s = [], []
  consistency_kls, consistency_rkls, consistency_skls = [], [], []

  i = 0
  while i < num_batches:
    try:
      samp = sess.run(z_sample)

      z_dist_p1, z_dist_p2, images, train_samp = sess.run(
          [dist_p1, dist_p2, input_images, z_sample_train],
          feed_dict={gan.z: samp})
      z_dist_p1s.append(z_dist_p1)
      z_dist_p2s.append(z_dist_p2)

      (consist_z_dist_p1_out, consist_z_dist_p2_out,
       consistency_images) = sess.run(
           [consist_z_dist_p1, consist_z_dist_p2, gan.fake_images],
           feed_dict={gan.z: train_samp})

      consist_z_dist_p1s.append(consist_z_dist_p1_out)
      consist_z_dist_p2s.append(consist_z_dist_p2_out)

      consist_kls, consist_rkls = sess.run(
          [consistency_kl, consistency_rkl],
          feed_dict={
              dist_p1_ph: z_dist_p1,
              dist_p2_ph: z_dist_p2,
              consist_dist_p1_ph: consist_z_dist_p1_out,
              consist_dist_p2_ph: consist_z_dist_p2_out,
          })

      consistency_kls.append(consist_kls)
      consistency_rkls.append(consist_rkls)
      consistency_skls.append((consist_kls + consist_rkls) / 2.0)

      if save_consistency_images:
        save_consistency_images -= 1

        filename = os.path.join(
            outdir,
            'consistency_image_%s_%06d_%06d.png'
            % (mode, i * gan.batch_size, (i + 1) * gan.batch_size - 1))
        img = consistency_images.reshape(
            [gan.batch_size * consistency_images.shape[1],
             consistency_images.shape[2],
             -1])
        _SaveImage(img, filename)

      if extract_input_images:
        extract_input_images -= 1

        if mode == 'self':
          filename = os.path.join(
              outdir,
              '%s_image_%06d_%06d.png'
              % (mode, i * gan.batch_size, (i + 1) * gan.batch_size - 1))
          img = images.reshape(
              [gan.batch_size * consistency_images.shape[1],
               consistency_images.shape[2],
               -1])
          _SaveImage(img, filename)
        else:
          for j in range(gan.batch_size):
            filename = os.path.join(
                outdir, '..', dataset,
                '%s_image_%06d.png' % (mode, i * gan.batch_size + j))
            _SaveImage(images[j], filename)

      if i % 100 == 0:
        tf.logging.info(
            '%s: step:%d consistency KL:%3.1f' %
            (mode, i, np.mean(consistency_skls)))

      i += 1
    except tf.errors.OutOfRangeError:
      break

  out_dists = dict(
      dist_p1=np.reshape(z_dist_p1s, [-1, gan.batch_size]),
      dist_p2=np.reshape(z_dist_p2s, [-1, gan.batch_size]),
      consist_dist_p1=np.reshape(consist_z_dist_p1s, [-1, gan.batch_size]),
      consist_dist_p2=np.reshape(consist_z_dist_p2s, [-1, gan.batch_size]),
      consistency_kl=np.reshape(consistency_kls, [-1, gan.batch_size]),
      consistency_rkl=np.reshape(consistency_rkls, [-1, gan.batch_size]),
      consistency_skl=np.reshape(consistency_skls, [-1, gan.batch_size]),
  )
  with tf.gfile.Open(
      os.path.join(outdir, '%s_consistency_dists.p' % mode), 'w') as f:
    pickle.dump(out_dists, f)

  return np.mean(consistency_skls)

