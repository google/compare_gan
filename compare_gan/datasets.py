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

"""Dataset loading utilities.

Creates a thin wrapper around TensorFlow Datasets (TFDS) to enable seamless
CPU/GPU/TPU workloads. The main entry point is 'get_dataset' which takes a
dataset name and a random seed and returns the corresponding tf.data.Dataset
object.

Available datasets are defined in the DATASETS dictionary. To add any dataset
supported by TFDS, simply extend the ImageDatasetV2 class as shown below with
the MNIST example and add it to DICTIONARY dictionary. Alternatively, you can
extend the ImageDatasetV2 class and load the datasets from another source.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect

from absl import flags
from absl import logging
from compare_gan.tpu import tpu_random
import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tfds_data_dir", None,
    "TFDS (TensorFlow Datasets) directory. If not set it will default to "
    "'~/tensorflow_datasets'. If the directory does not contain the requested "
    "dataset TFDS will download the dataset to this folder.")

flags.DEFINE_boolean(
    "data_fake_dataset", False,
    "If True don't load datasets from disk but create fake values.")

flags.DEFINE_integer(
    "data_shuffle_buffer_size", 10000,
    "Number of examples for the shuffle buffer.")

# Deprecated, only used for "replacing labels". TFDS will always use 64 threads.
flags.DEFINE_integer(
    "data_reading_num_threads", 64,
    "The number of threads used to read the dataset.")


class ImageDatasetV2(object):
  """Interface for Image datasets based on TFDS (TensorFlow Datasets).

  This method handles both CPU/GPU and TPU data loading settings. If the flag
  --data_fake_dataset is True the methods will create a small fake dataset from
  in-memory NumPy arrays and not read from disk.
  The pipleline of input operations is as follows:
  1) Shuffle filenames (with seed).
  2) Load file content from disk. Decode images.
  Dataset content after this step is a dictionary.
  3) Prefetch call here.
  4) Filter examples (e.g. by size or label).
  5) Parse example.
  Dataset content after this step is a tuple of tensors (image, label).
  6) train_only: Repeat dataset.
  7) Transform (random cropping with seed, resizing).
  8) Preprocess (adding sampled noise/labels with seed).
  Dataset content after this step is a tuple (feature dictionary, label tensor).
  9) train only: Shuffle examples (with seed).
  10) Batch examples.
  11) Prefetch examples.

  Step 1-3 are done by _load_dataset() and wrap tfds.load().
  Step 4-11 are done by train_input_fn() and eval_input_fn().
  """

  def __init__(self,
               name,
               tfds_name,
               resolution,
               colors,
               num_classes,
               eval_test_samples,
               seed):
    logging.info("ImageDatasetV2(name=%s, tfds_name=%s, resolution=%d, "
                 "colors=%d, num_classes=%s, eval_test_samples=%s, seed=%s)",
                 name, tfds_name, resolution, colors, num_classes,
                 eval_test_samples, seed)
    self._name = name
    self._tfds_name = tfds_name
    self._resolution = resolution
    self._colors = colors
    self._num_classes = num_classes
    self._eval_test_sample = eval_test_samples
    self._seed = seed

    self._train_split = tfds.Split.TRAIN
    self._eval_split = tfds.Split.TEST

  @property
  def name(self):
    """Name of the dataset."""
    return self._name

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def eval_test_samples(self):
    """Number of examples in the "test" split of this dataset."""
    if FLAGS.data_fake_dataset:
      return 100
    return self._eval_test_sample

  @property
  def image_shape(self):
    """Returns a tuple with the image shape."""
    return (self._resolution, self._resolution, self._colors)

  def _make_fake_dataset(self, split):
    """Returns a fake data set with the correct shapes."""
    np.random.seed(self._seed)
    num_samples_per_epoch = 100
    num_epochs = self.eval_test_samples // 100 if split == "test" else None
    images_shape = [num_samples_per_epoch] + list(self.image_shape)
    images = np.random.uniform(size=images_shape).astype(np.float32)
    labels = np.ones((num_samples_per_epoch,), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    return ds.repeat(num_epochs)

  def _get_per_host_random_seed(self, tpu_context=None):
    """Returns the dataset seed for according to the TPUContext.

    On CPU/GPU it returns the default seed. For TPUs the input_fn is executed
    on every host machine (if per-host input is set, which is set by default).
    We use a different (but deterministically computed) random seed on each host
    to ensure each host machine sees a different stream of input data.

    Args:
      tpu_context: TPU execution context.

    Returns:
      The current seed if CPU/GPU and a host-specific seed for TPU.
    """
    if self._seed is None:
      logging.warning("Dataset seed not set.")
      return None
    if tpu_context is None:
      logging.warning("No TPUContext, using unmodified dataset seed %s.",
                      self._seed)
      return self._seed
    seed = self._seed + tpu_context.current_host
    logging.info("Running with %d hosts, modifying dataset seed for "
                 "host %d to %s.", tpu_context.num_hosts,
                 tpu_context.current_host, seed)
    return seed

  @gin.configurable("replace_labels", whitelist=["file_pattern"])
  def _replace_labels(self, split, ds, file_pattern=None):
    """Replaces the labels in the dataset with labels from separate files.

    This functionality is used if one wants to either replace the labels with
    soft labels (i.e. softmax over the logits) or label the instances with
    a new classifier.

    Args:
      split: Dataset split (e.g. train/test/validation).
      ds: The underlying TFDS object.
      file_pattern: Path to the replacement files.

    Returns:
      An instance of tf.data.Dataset with the updated labels.
    """
    if not file_pattern:
      return ds
    file_pattern = file_pattern.format(split=split)
    logging.warning("Using labels from %s for split %s.", file_pattern, split)
    label_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    label_ds = label_ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=FLAGS.data_reading_num_threads)
    ds = tf.data.Dataset.zip((ds, label_ds)).map(self._replace_label)
    return ds

  def _replace_label(self, feature_dict, new_unparsed_label):
    """Replaces the label from the feature_dict with the new label.

    Furthermore, if the feature_dict contains a key for the file_name which
    identifies an instance, we double-check that the we are replacing the label
    of the correct instance.

    Args:
      feature_dict: A serialized TFRecord containing the old label.
      new_unparsed_label: A serialized TFRecord containing the new label.

    Returns:
      Updates the label in the label dict to the new label.
    """
    label_spec = {
        "file_name": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.int64),
    }
    parsed_label = tf.parse_single_example(new_unparsed_label, label_spec)
    with tf.control_dependencies([
        tf.assert_equal(parsed_label["file_name"], feature_dict["file_name"])]):
      feature_dict["label"] = tf.identity(parsed_label["label"])
    return feature_dict

  def _parse_fn(self, features):
    image = tf.cast(features["image"], tf.float32) / 255.0
    return image, features["label"]

  def _load_dataset(self, split):
    """Loads the underlying dataset split from disk.

    Args:
      split: Name of the split to load.

    Returns:
      Returns a `tf.data.Dataset` object with a tuple of image and label tensor.
    """
    if FLAGS.data_fake_dataset:
      return self._make_fake_dataset(split)
    ds = tfds.load(
        self._tfds_name,
        split=split,
        data_dir=FLAGS.tfds_data_dir,
        as_dataset_kwargs={"shuffle_files": False})
    ds = self._replace_labels(split, ds)
    ds = ds.map(self._parse_fn)
    return ds.prefetch(tf.contrib.data.AUTOTUNE)

  def _train_filter_fn(self, image, label):
    del image, label
    return True

  def _train_transform_fn(self, image, label, seed):
    del seed
    return image, label

  def _eval_transform_fn(self, image, label, seed):
    del seed
    return image, label

  def train_input_fn(self, params=None, preprocess_fn=None):
    """Input function for reading data.

    Args:
      params: Python dictionary with parameters. Must contain the key
        "batch_size". TPUEstimator will set this for you!
      preprocess_fn: Function to process single examples. This is allowed to
        have a `seed` argument.

    Returns:
      `tf.data.Dataset` with preprocessed and batched examples.
    """
    if params is None:
      params = {}
    seed = self._get_per_host_random_seed(params.get("context", None))
    logging.info("train_input_fn(): params=%s seed=%s", params, seed)

    ds = self._load_dataset(split=self._train_split)
    ds = ds.filter(self._train_filter_fn)
    ds = ds.repeat()
    ds = ds.map(functools.partial(self._train_transform_fn, seed=seed))
    if preprocess_fn is not None:
      if "seed" in inspect.getargspec(preprocess_fn).args:
        preprocess_fn = functools.partial(preprocess_fn, seed=seed)
      ds = ds.map(preprocess_fn)
      # Add a feature for the random offset of operations in tpu_random.py.
      ds = tpu_random.add_random_offset_to_features(ds)
    ds = ds.shuffle(FLAGS.data_shuffle_buffer_size, seed=seed)
    if "batch_size" in params:
      ds = ds.batch(params["batch_size"], drop_remainder=True)
    return ds.prefetch(tf.contrib.data.AUTOTUNE)

  def eval_input_fn(self, params=None, split=None):
    """Input function for reading data.

    Args:
      params: Python dictionary with parameters. Must contain the key
        "batch_size". TPUEstimator will set this for you!
      split: Name of the split to use. If None will use the default eval split
        of the dataset.

    Returns:
      `tf.data.Dataset` with preprocessed and batched examples.
    """
    if params is None:
      params = {}
    if split is None:
      split = self._eval_split
    seed = self._get_per_host_random_seed(params.get("context", None))
    logging.info("eval_input_fn(): params=%s seed=%s", params, seed)

    ds = self._load_dataset(split=split)
    # No filter, no rpeat.
    ds = ds.map(functools.partial(self._eval_transform_fn, seed=seed))
    # No shuffle.
    if "batch_size" in params:
      ds = ds.batch(params["batch_size"], drop_remainder=True)
    return ds.prefetch(tf.contrib.data.AUTOTUNE)

  # For backwards compatibility ImageDataset.
  def input_fn(self, params, mode=tf.estimator.ModeKeys.TRAIN,
               preprocess_fn=None):
    assert mode == tf.estimator.ModeKeys.TRAIN, mode
    return self.train_input_fn(params=params, preprocess_fn=preprocess_fn)

  # For backwards compatibility ImageDataset.
  def load_dataset(self, split_name):
    assert split_name == "test", split_name
    return self.eval_input_fn()


class MnistDataset(ImageDatasetV2):
  """Wrapper for the MNIST dataset from TFDS."""

  def __init__(self, seed):
    super(MnistDataset, self).__init__(
        name="mnist",
        tfds_name="mnist",
        resolution=28,
        colors=1,
        num_classes=10,
        eval_test_samples=10000,
        seed=seed)


class FashionMnistDataset(ImageDatasetV2):
  """Wrapper for the Fashion-MNIST dataset from TDFS."""

  def __init__(self, seed):
    super(FashionMnistDataset, self).__init__(
        name="fashion_mnist",
        tfds_name="fashion_mnist",
        resolution=28,
        colors=1,
        num_classes=10,
        eval_test_samples=10000,
        seed=seed)


class Cifar10Dataset(ImageDatasetV2):
  """Wrapper for the CIFAR10 dataset from TDFS."""

  def __init__(self, seed):
    super(Cifar10Dataset, self).__init__(
        name="cifar10",
        tfds_name="cifar10",
        resolution=32,
        colors=3,
        num_classes=10,
        eval_test_samples=10000,
        seed=seed)


class CelebaDataset(ImageDatasetV2):
  """Wrapper for the CelebA dataset from TFDS."""

  def __init__(self, seed):
    super(CelebaDataset, self).__init__(
        name="celeb_a",
        tfds_name="celeb_a",
        resolution=64,
        colors=3,
        num_classes=None,
        eval_test_samples=10000,
        seed=seed)

  def _parse_fn(self, features):
    """Returns 64x64x3 image and constant label."""
    image = features["image"]
    image = tf.image.resize_image_with_crop_or_pad(image, 160, 160)
    # Note: possibly consider using NumPy's imresize(image, (64, 64))
    image = tf.image.resize_images(image, [64, 64])
    image.set_shape(self.image_shape)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.constant(0, dtype=tf.int32)
    return image, label


class LsunBedroomDataset(ImageDatasetV2):
  """Wrapper from the LSUN Bedrooms dataset from TFDS."""

  def __init__(self, seed):
    super(LsunBedroomDataset, self).__init__(
        name="lsun-bedroom",
        tfds_name="lsun/bedroom",
        resolution=128,
        colors=3,
        num_classes=None,
        eval_test_samples=30000,
        seed=seed)

    # As the official LSUN validation set only contains 300 samples, which is
    # insufficient for FID computation, we're splitting off some trianing
    # samples. The smallest percentage selectable through TFDS is 1%, so we're
    # going to use that (corresponding roughly to 30000 samples).
    # If you want to use fewer eval samples, just modify eval_test_samples.
    self._train_split, self._eval_split = \
        tfds.Split.TRAIN.subsplit([99, 1])

  def _parse_fn(self, features):
    """Returns a 128x128x3 Tensor with constant label 0."""
    image = features["image"]
    image = tf.image.resize_image_with_crop_or_pad(
        image, target_height=128, target_width=128)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.constant(0, dtype=tf.int32)
    return image, label


def _transform_imagnet_image(image, target_image_shape, crop_method, seed):
  """Preprocesses ImageNet images to have a target image shape.

  Args:
    image: 3-D tensor with a single image.
    target_image_shape: List/Tuple with target image shape.
    crop_method: Method for cropping the image:
      One of: distorted, random, middle, none
    seed: Random seed, only used for `crop_method=distorted`.

  Returns:
    Image tensor with shape `target_image_shape`.
  """
  if crop_method == "distorted":
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        aspect_ratio_range=[1.0, 1.0],
        area_range=[0.5, 1.0],
        use_image_if_no_bounding_boxes=True,
        seed=seed)
    image = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    image.set_shape([None, None, target_image_shape[-1]])
  elif crop_method == "random":
    tf.set_random_seed(seed)
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = [h - size, w - size] * tf.random.uniform([2], 0, 1)
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method == "middle":
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([h - size, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method != "none":
    raise ValueError("Unsupported crop method: {}".format(crop_method))
  image = tf.image.resize_images(
      image, [target_image_shape[0], target_image_shape[1]])
  image.set_shape(target_image_shape)
  return image


@gin.configurable("train_imagenet_transform", whitelist=["crop_method"])
def _train_imagenet_transform(image, target_image_shape, seed,
                              crop_method="distorted"):
  return _transform_imagnet_image(
      image,
      target_image_shape=target_image_shape,
      crop_method=crop_method,
      seed=seed)


@gin.configurable("eval_imagenet_transform", whitelist=["crop_method"])
def _eval_imagenet_transform(image, target_image_shape, seed,
                             crop_method="middle"):
  return _transform_imagnet_image(
      image,
      target_image_shape=target_image_shape,
      crop_method=crop_method,
      seed=seed)


class ImagenetDataset(ImageDatasetV2):
  """ImageNet2012 as defined by TF Datasets."""

  def __init__(self, resolution, seed, filter_unlabeled=False):
    if resolution not in [64, 128, 256, 512]:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    super(ImagenetDataset, self).__init__(
        name="imagenet_{}".format(resolution),
        tfds_name="imagenet2012",
        resolution=resolution,
        colors=3,
        num_classes=1000,
        eval_test_samples=50000,
        seed=seed)
    self._eval_split = tfds.Split.VALIDATION
    self._filter_unlabeled = filter_unlabeled

  def _train_filter_fn(self, image, label):
    del image
    if not self._filter_unlabeled:
      return True
    logging.warning("Filtering unlabeled examples.")
    return tf.math.greater_equal(label, 0)

  def _train_transform_fn(self, image, label, seed):
    image = _train_imagenet_transform(
        image=image, target_image_shape=self.image_shape, seed=seed)
    return image, label

  def _eval_transform_fn(self, image, label, seed):
    image = _eval_imagenet_transform(
        image=image, target_image_shape=self.image_shape, seed=seed)
    return image, label


class SizeFilteredImagenetDataset(ImagenetDataset):
  """ImageNet from TFDS filtered by image size."""

  def __init__(self, resolution, threshold, seed):
    super(SizeFilteredImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "imagenet_{}_hq{}".format(resolution, threshold)
    self._threshold = threshold

  def _train_filter_fn(self, image, label):
    """The minimum image dimension has to be larger than the threshold."""
    del label
    size = tf.math.reduce_min(tf.shape(image)[:2])
    return tf.greater_equal(size, self._threshold)


class SingleClassImagenetDataset(ImagenetDataset):
  """ImageNet from TFDS with all instances having a constant label 0.

  It can be used to simmulate the setting where no labels are provided.
  """

  def __init__(self, resolution, seed):
    super(SingleClassImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "single_class_" + self._name
    self._num_classes = 1

  def _parse_fn(self, features):
    image, _ = super(SingleClassImagenetDataset, self)._parse_fn(features)
    label = tf.constant(0, dtype=tf.int32)
    return image, label


class RandomClassImagenetDataset(ImagenetDataset):
  """ImageNet2012 dataset with random labels."""

  def __init__(self, resolution, seed):
    super(RandomClassImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "random_class_" + self._name
    self._num_classes = 1000

  def _parse_fn(self, features):
    image, _ = super(RandomClassImagenetDataset, self)._parse_fn(features)
    label = tf.random.uniform(minval=0, maxval=1000, dtype=tf.int32)
    return image, label


class SoftLabeledImagenetDataset(ImagenetDataset):
  """ImageNet2012 dataset with soft labels."""

  def __init__(self, resolution, seed):
    super(SoftLabeledImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "soft_labeled_" + self._name

  def _replace_label(self, feature_dict, new_unparsed_label):
    """Replaces the label from the feature_dict with the new (soft) label.

    The function assumes that the new_unparsed_label contains a list of logits
    which will be converted to a soft label using the softmax.

    Args:
      feature_dict: A serialized TFRecord containing the old label.
      new_unparsed_label: A serialized TFRecord containing the new label.

    Returns:
      Updates the label in the label dict to the new soft label.
    """
    label_spec = {
        "file_name": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature([self._num_classes], tf.float32)
    }
    parsed_label = tf.parse_single_example(new_unparsed_label, label_spec)
    with tf.control_dependencies([
        tf.assert_equal(parsed_label["file_name"], feature_dict["file_name"])]):
      feature_dict["label"] = tf.nn.softmax(logits=parsed_label["label"])
    return feature_dict


DATASETS = {
    "celeb_a": CelebaDataset,
    "cifar10": Cifar10Dataset,
    "fashion-mnist": FashionMnistDataset,
    "lsun-bedroom": LsunBedroomDataset,
    "mnist": MnistDataset,
    "imagenet_64": functools.partial(ImagenetDataset, resolution=64),
    "imagenet_128": functools.partial(ImagenetDataset, resolution=128),
    "imagenet_256": functools.partial(ImagenetDataset, resolution=256),
    "imagenet_512": functools.partial(ImagenetDataset, resolution=512),
    "imagenet_512_hq400": (functools.partial(
        SizeFilteredImagenetDataset, resolution=512, threshold=400)),
    "soft_labeled_imagenet_128": functools.partial(
        SoftLabeledImagenetDataset, resolution=128),
    "single_class_imagenet_128": functools.partial(
        SingleClassImagenetDataset, resolution=128),
    "random_class_imagenet_128": functools.partial(
        RandomClassImagenetDataset, resolution=128),
    "labeled_only_imagenet_128": functools.partial(
        ImagenetDataset, resolution=128, filter_unlabeled=True),
}


@gin.configurable("dataset")
def get_dataset(name, seed=547):
  """Instantiates a data set and sets the random seed."""
  if name not in DATASETS:
    raise ValueError("Dataset %s is not available." % name)
  return DATASETS[name](seed=seed)
