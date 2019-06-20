# Compare GAN

This repository offers TensorFlow implementations for many components related to
**Generative Adversarial Networks**:

*   losses (such non-saturating GAN, least-squares GAN, and WGAN),
*   penalties (such as the gradient penalty),
*   normalization techniques (such as spectral normalization, batch
    normalization, and layer normalization),
*   neural architectures (BigGAN, ResNet, DCGAN), and
*   evaluation metrics (FID score, Inception Score, precision-recall, and KID
    score).

The code is **configurable via [Gin](https://github.com/google/gin-config)** and
runs on **GPU/TPU/CPUs**. Several research papers make use of this repository,
including:

1.  [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan/tree/v1)
    \
    Mario Lucic*, Karol Kurach*, Marcin Michalski, Sylvain Gelly, Olivier
    Bousquet **[NeurIPS 2018]**

2.  [The GAN Landscape: Losses, Architectures, Regularization, and Normalization](https://arxiv.org/abs/1807.04720)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan/tree/v2)
    [<font color="green">[Colab]</font>](https://colab.research.google.com/github/google/compare_gan/blob/v2/compare_gan/src/tfhub_models.ipynb)
    \
    Karol Kurach*, Mario Lucic*, Xiaohua Zhai, Marcin Michalski, Sylvain Gelly
    **[ICML 2019]**

3.  [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan/blob/560697ee213f91048c6b4231ab79fcdd9bf20381/compare_gan/src/prd_score.py)
    \
    Mehdi S. M. Sajjadi, Olivier Bachem, Mario Lucic, Olivier Bousquet, Sylvain
    Gelly **[NeurIPS 2018]**

4.  [GILBO: One Metric to Measure Them All](https://arxiv.org/abs/1802.04874)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan/blob/560697ee213f91048c6b4231ab79fcdd9bf20381/compare_gan/src/gilbo.py)
    \
    Alexander A. Alemi, Ian Fischer **[NeurIPS 2018]**

5.  [A Case for Object Compositionality in Deep Generative Models of Images](https://arxiv.org/abs/1810.10340)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan/tree/v2_multigan)
    \
    Sjoerd van Steenkiste, Karol Kurach, Sylvain Gelly **[2018]**

6.  [On Self Modulation for Generative Adversarial Networks](https://arxiv.org/abs/1810.01365)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan) \
    Ting Chen, Mario Lucic, Neil Houlsby, Sylvain Gelly **[ICLR 2019]**

7.  [Self-Supervised GANs via Auxiliary Rotation Loss](https://arxiv.org/abs/1811.11212)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan)
    [<font color="green">[Colab]</font>](https://colab.research.google.com/github/google/compare_gan/blob/v3/colabs/ssgan_demo.ipynb)
    \
    Ting Chen, Xiaohua Zhai, Marvin Ritter, Mario Lucic, Neil Houlsby **[CVPR
    2019]**

8.  [High-Fidelity Image Generation With Fewer Labels](https://arxiv.org/abs/1903.02271)
    [<font color="green">[Code]</font>](https://github.com/google/compare_gan)
    [<font color="green">[Blog Post]</font>](https://ai.googleblog.com/2019/03/reducing-need-for-labeled-data-in.html)
    [<font color="green">[Colab]</font>](https://colab.research.google.com/github/google/compare_gan/blob/v3/colabs/s3gan_demo.ipynb)
    \
    Mario Lucic*, Michael Tschannen*, Marvin Ritter*, Xiaohua Zhai, Olivier
    Bachem, Sylvain Gelly **[ICML 2019]**

## Installation

You can easily install the library and all necessary dependencies by running:
`pip install -e .` from the `compare_gan/` folder.

## Running experiments

Simply run the `main.py` passing a `--model_dir` (this is where checkpoints are
stored) and a `--gin_config` (defines which model is trained on which data set
and other training options). We provide several example configurations in the
`example_configs/` folder:

*   **dcgan_celeba64**: DCGAN architecture with non-saturating loss on CelebA
    64x64px
*   **resnet_cifar10**: ResNet architecture with non-saturating loss and
    spectral normalization on CIFAR-10
*   **resnet_lsun-bedroom128**: ResNet architecture with WGAN loss and gradient
    penalty on LSUN-bedrooms 128x128px
*   **sndcgan_celebahq128**: SN-DCGAN architecture with non-saturating loss and
    spectral normalization on CelebA-HQ 128x128px
*   **biggan_imagenet128**: BigGAN architecture with hinge loss and spectral
    normalization on ImageNet 128x128px

### Training and evaluation

To see all available options please run `python main.py --help`. Main options:

*   To **train** the model use `--schedule=train` (default). Training is resumed
    from the last saved checkpoint.
*   To **evaluate** all checkpoints use `--schedule=continuous_eval
    --eval_every_steps=0`. To evaluate only checkpoints where the step size is
    divisible by 5000, use `--schedule=continuous_eval --eval_every_steps=5000`.
    By default, 3 averaging runs are used to estimate the Inception Score and
    the FID score. Keep in mind that when running locally on a single GPU it may
    not be possible to run training and evaluation simultaneously due to memory
    constraints.
*   To **train and evaluate** the model use `--schedule=eval_after_train
    --eval_every_steps=0`.

### Training on Cloud TPUs

We recommend using the
[ctpu tool](https://github.com/tensorflow/tpu/tree/master/tools/ctpu) to create
a Cloud TPU and corresponding Compute Engine VM. We use v3-128 Cloud TPU v3 Pod
for training models on ImageNet in 128x128 resolutions. You can use smaller
slices if you reduce the batch size (`options.batch_size` in the Gin config) or
model parameters. Keep in mind that the model quality might change. Before
training make sure that the environment variable `TPU_NAME` is set. Running
evaluation on TPUs is currently not supported. Use a VM with a single GPU
instead.

### Datasets

Compare GAN uses [TensorFlow Datasets](https://www.tensorflow.org/datasets) and
it will automatically download and prepare the data. For ImageNet you will need
to download the archive yourself. For CelebAHq you need to download and prepare
the images on your own. If you are using TPUs make sure to point the training
script to your Google Storage Bucket (`--tfds_data_dir`).
