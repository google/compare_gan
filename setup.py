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

"""Install compare_gan."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='compare_gan',
    version='1.0',
    description=('Compare GAN - code from "Are GANs created equal? '
                 'A Large Scale Study" paper'),
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/TODO',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
    },
    scripts=[
        'compare_gan/bin/compare_gan_generate_tasks',
        'compare_gan/bin/compare_gan_prepare_datasets.sh',
        'compare_gan/bin/compare_gan_run_one_task',
        'compare_gan/bin/compare_gan_run_test.sh',
    ],
    install_requires=[
        'future',
        'numpy',
        'pandas',
        'protobuf',
        'six',
        'tensor2tensor',
    ],
    extras_require={
        'matplotlib': ['matplotlib>=1.5.2'],
        'pillow': ['pillow>=5.0.0'],
        'pandas': ['pandas>=0.23.0'],
        'pstar': ['pstar>=0.1.6'],
        'scipy': ['scipy>=1.0.0'],
        'tensorflow': ['tensorflow>=1.7'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.4.1'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning gan',
)
