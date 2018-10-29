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

rm -rf /tmp/results
compare_gan_generate_tasks --workdir=/tmp/results --experiment=test
compare_gan_run_one_task --workdir=/tmp/results --task_num=0 --alsologtostderr --dataset_root=/tmp/datasets

echo "Training model and evaluating GILBO score"

compare_gan_generate_tasks --workdir=/tmp/results_gilbo --experiment=test_gilbo
compare_gan_run_one_task --workdir=/tmp/results_gilbo --task_num=0 --alsologtostderr --dataset_root=/tmp/datasets
