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

"""Defines constants used across the code base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


NORMAL_INIT = "normal"
TRUNCATED_INIT = "truncated"
ORTHOGONAL_INIT = "orthogonal"
INITIALIZERS = [NORMAL_INIT, TRUNCATED_INIT, ORTHOGONAL_INIT]

NO_NORMALIZATION = "none"
BATCH_NORM = "batch_norm"
LAYER_NORM = "layer_norm"
SPECTRAL_NORM = "spectral_norm"
WEIGHT_NORM = "weight_norm"
NORMALIZERS = [NO_NORMALIZATION, BATCH_NORM, LAYER_NORM,
               SPECTRAL_NORM, WEIGHT_NORM]

INFOGAN_ARCH = "infogan_arch"
DCGAN_ARCH = "dcgan_arch"
RESNET5_ARCH = "resnet5_arch"
RESNET5_BIGGAN_ARCH = "resnet5_biggan_arch"
SNDCGAN_ARCH = "sndcgan_arch"
RESNET_CIFAR = "resnet_cifar_arch"
RESNET_STL = "resnet_stl_arch"
RESNET30_ARCH = "resnet30_arch"
RESNET5_ABLATION = "resnet5_ablation"  # Only to compare changes in Resnet.
ARCHITECTURES = [INFOGAN_ARCH, DCGAN_ARCH, RESNET_CIFAR, SNDCGAN_ARCH,
                 RESNET5_ARCH, RESNET5_BIGGAN_ARCH, RESNET30_ARCH, RESNET_STL,
                 RESNET5_ABLATION]

# This will be used when checking that the regularization was applied
# on a correct number of layers. Currently used only for L2 regularization in
# penalty_lib.
N_DISCRIMINATOR_LAYERS = {
    INFOGAN_ARCH: 4,
    DCGAN_ARCH: 5,
    SNDCGAN_ARCH: 8,
    RESNET_CIFAR: 13,
    RESNET5_ARCH: 19,
    RESNET5_BIGGAN_ARCH: 19,
    RESNET30_ARCH: 107,
    RESNET5_ABLATION: 19,
}
