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

#!/bin/bash

T2T_DATAGEN="$HOME/.local/bin/t2t-datagen"
DATASET_DIR="/tmp/datasets"
TMP_DIR="/tmp"

if [ ! -f ${T2T_DATAGEN?} ]; then
  echo "tensor2tensor not found!"
  exit 1
fi

echo "Preparing datasets (mnist, fashionmnist, cifar10, celeba) in ${DATASET_DIR?}"
mkdir ${DATASET_DIR?}

datasets=(image_mnist image_fashion_mnist image_cifar10 image_celeba)
for dataset in "${datasets[@]}"; do
  echo "Dataset is ${dataset?}"
  ${T2T_DATAGEN?} --data_dir=${DATASET_DIR?} --problem=${dataset?} --tmp_dir=${TMP_DIR?}
done

echo "Getting inception model."
(cd ${DATASET_DIR?}; curl "http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz" | tar -zxvf - inceptionv1_for_inception_score.pb )
