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
    version='3.0',
    description=(
        'Compare GAN - A modular library for training and evaluating GANs.'),
    author='Google LLC',
    author_email='no-reply@google.com',
    url='https://github.com/google/compare_gan',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={},
    install_requires=[
        'future',
        'gin-config==0.1.4',
        'numpy',
        'pandas',
        'six',
        'tensorflow-datasets==1.0.1',
        'tensorflow-hub>=0.2.0',
        'tensorflow-gan==0.0.0.dev0',
        'matplotlib>=1.5.2',
        'pstar>=0.1.6',
        'scipy>=1.0.0',
    ],
    extras_require={
        'tf': ['tensorflow>=1.12'],
        # Evaluation of Hub modules with EMA variables requires TF > 1.12.
        'tf_gpu': ['tf-nightly-gpu>=1.13.0.dev20190221'],
        'pillow': ['pillow>=5.0.0'],
        'tensorflow-probability': ['tensorflow-probability>=0.5.0'],
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
