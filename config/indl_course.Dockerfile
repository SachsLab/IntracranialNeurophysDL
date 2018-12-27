# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda:9.0-base-ubuntu${UBUNTU_VERSION}
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        unzip \
        bzip2 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The following is not in the official tensorflow dockerfile but it is in every other tensorflow dockerfile I see.
ENV CUDA_HOME /usr/local/cuda
# export CUDA_HOME=/usr/local/cuda

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
# export LANG=C.UTF-8

# We will download supporting tools into the HOME (/root) directory.
WORKDIR /root

# Install conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh
ENV PATH /root/miniconda/bin:$PATH
# export PATH=/root/miniconda/bin:$PATH
RUN . /root/miniconda/etc/profile.d/conda.sh
RUN conda update -n base -c defaults conda

# Download the course repo so we have access to the conda environment.yml
# Username and password are required until I make the repo public.
ARG GHUSER
ARG GHPASS
RUN git clone https://${GHUSER}:${GHPASS}@github.com/SachsLab/IntracranialNeurophysDL.git ~/indl_repo

# Create a new conda environment. All the details come from the provided environment.yml.
# It takes a while to solve the dependencies and then finally do the install.
RUN conda env create -n indl -f /root/indl_repo/config/environment.yml
ENV PATH /root/miniconda/envs/indl/bin:$PATH
# export PATH=/root/miniconda/envs/indl/bin:$PATH
# Test in a --runtime=nvidia session with:
# python -c "import tensorflow as tf; tf.test.is_gpu_available()"

# Download fastai and install from its setup.py
RUN git clone --depth=1 https://github.com/fastai/fastai ~/fastai
RUN pip install ~/fastai/
# Also install torchtext. We could do it in environment.yml but its dependencies clash with fastai dependencies.
RUN pip install torchtext
# Test in a --runtime=nvidia session with:
# python -c "import torch; print(torch.rand(2,3).cuda())"

# Download learning materials from tensorflow docs, fastai course.
RUN git clone --depth=1 https://github.com/tensorflow/docs.git ~/tf_docs
RUN git clone https://github.com/fastai/course-v3.git ~/fastai_v3

# Link the course notebooks into /notebooks
WORKDIR /notebooks
RUN chmod -R a+w /notebooks
RUN ln -fs ~/tf_docs/site/en /notebooks/tensorflow
RUN ln -fs ~/fastai/courses /notebooks/fastaiv2
RUN ln -fs ~/fastai_v3/nbs /notebooks/fastaiv3
RUN ln -fs ~/indl_repo/lessons /notebooks/indl

# The course notebooks expect data to be stored in a specific location.
# When the docker container is run, it has mapped persistent storage. e.g. `-v $PWD/data:/root/data`
# Here we make sure that folder exists, and we link it to the expected data locations.
WORKDIR /root/data
WORKDIR /root/.keras
WORKDIR /root/.fastai
WORKDIR /root/.torch
RUN ln -fs /root/data /root/.keras/datasets
RUN ln -fs /root/data /root/.fastai/data
RUN ln -fs /root/data /notebooks/indl/data

# Set the working directory for when the container is run.
WORKDIR /notebooks

# Run the jupyter notebook
EXPOSE 8888
CMD ["bash", "-c", "source ~/.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://127.0.0.1:8888'"]
