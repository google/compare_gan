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

DCGAN_ARCH = "dcgan_arch"
DUMMY_ARCH = "dummy_arch"
INFOGAN_ARCH = "infogan_arch"
RESNET5_ARCH = "resnet5_arch"
RESNET30_ARCH = "resnet30_arch"
RESNET_BIGGAN_ARCH = "resnet_biggan_arch"
RESNET_BIGGAN_DEEP_ARCH = "resnet_biggan_deep_arch"
RESNET_CIFAR_ARCH = "resnet_cifar_arch"
RESNET_STL_ARCH = "resnet_stl_arch"
SNDCGAN_ARCH = "sndcgan_arch"
ARCHITECTURES = [INFOGAN_ARCH, DCGAN_ARCH, RESNET5_ARCH, RESNET30_ARCH,
                 RESNET_BIGGAN_ARCH, RESNET_BIGGAN_DEEP_ARCH, RESNET_CIFAR_ARCH,
                 RESNET_STL_ARCH, SNDCGAN_ARCH]
