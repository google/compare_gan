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


GAN_WITH_PENALTY = "GAN_PENALTY"
WGAN_WITH_PENALTY = "WGAN_PENALTY"
LSGAN_WITH_PENALTY = "LSGAN_PENALTY"
MODELS_WITH_PENALTIES = [GAN_WITH_PENALTY, WGAN_WITH_PENALTY,
                         LSGAN_WITH_PENALTY]

NO_NORMALIZATION = "none"
BATCH_NORM = "batch_norm"
LAYER_NORM = "layer_norm"
SPECTRAL_NORM = "spectral_norm"
WEIGHT_NORM = "weight_norm"
NORMALIZERS = [NO_NORMALIZATION, BATCH_NORM, LAYER_NORM,
               SPECTRAL_NORM, WEIGHT_NORM]

NO_PENALTY = "no_penalty"
WGANGP_PENALTY = "wgangp_penalty"
DRAGAN_PENALTY = "dragan_penalty"
L2_PENALTY = "l2_penalty"
PENALTIES = [NO_PENALTY, WGANGP_PENALTY, DRAGAN_PENALTY, L2_PENALTY]

INFOGAN_ARCH = "infogan_arch"
DCGAN_ARCH = "dcgan_arch"
RESNET5_ARCH = "resnet5_arch"
SNDCGAN_ARCH = "sndcgan_arch"
RESNET_CIFAR = "resnet_cifar_arch"
RESNET_STL = "resnet_stl_arch"
RESNET107_ARCH = "resnet107_arch"
RESNET5_ABLATION = "resnet5_ablation"  # Only to compare changes in Resnet.
ARCHITECTURES = [INFOGAN_ARCH, DCGAN_ARCH, RESNET_CIFAR, SNDCGAN_ARCH,
                 RESNET5_ARCH, RESNET107_ARCH, RESNET_STL, RESNET5_ABLATION]

# This will be used when checking that the regularization was applied
# on a correct number of layers. Currently used only for L2 regularization.

N_DISCRIMINATOR_LAYERS = {
    INFOGAN_ARCH: 4,
    DCGAN_ARCH: 5,
    SNDCGAN_ARCH: 8,
    RESNET_CIFAR: 13,
    RESNET5_ARCH: 19,
    RESNET107_ARCH: 107,
    RESNET5_ABLATION: 19,
}
