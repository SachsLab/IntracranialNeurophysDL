ARG UBUNTU_MAJOR=18
ARG UBUNTU_MINOR=04
ARG ARCH=
ARG CUDA=10.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-cudnn7-runtime-ubuntu${UBUNTU_MAJOR}.${UBUNTU_MINOR} as base

# Some ARGS are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG UBUNTU_MAJOR
ARG UBUNTU_MINOR
ARG ARCH
ARG CUDA
ARG CUDNN=7.4.1.5-1

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
# export LANG=C.UTF-8

# Pick up some TF dependencies and other tools we will use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
#        cuda-command-line-tools-${CUDA/./-} \
#        cuda-cublas-${CUDA/./-} \
#        cuda-cufft-${CUDA/./-} \
#        cuda-curand-${CUDA/./-} \
#        cuda-cusolver-${CUDA/./-} \
#        cuda-cusparse-${CUDA/./-} \
#        libcudnn7=${CUDNN}+cuda${CUDA} \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        wget \
        git \
        unzip \
        bzip2 && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu${UBUNTU_MAJOR}${UBUNTU_MINOR}-5.0.2-ga-cuda${CUDA} \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*)

# The following is not in the official tensorflow dockerfile but it is in every other tensorflow dockerfile I see.
ENV CUDA_HOME /usr/local/cuda
# export CUDA_HOME=/usr/local/cuda

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ARG PYTHON=python3.6
ARG PIP=pip3
RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    python3-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tf-nightly-gpu
ARG TF_PACKAGE_VERSION=
RUN pip3 install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

RUN chmod a+rwx /etc/bash.bashrc

RUN pip3 install jupyter matplotlib
RUN pip3 install jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

# We will download supporting tools into the HOME (/root) directory.
WORKDIR /root

RUN pip3 install fastai opencv-python seaborn graphviz scikit-learn ipywidgets

RUN mkdir -p /root/.torch/models
RUN mkdir -p /root/.fastai/data

# install jaxlib
# JAX_PY_VER alternatives: cp27, cp35, cp36, cp37
ARG JAX_PY_VER=cp36
# JAX_CUDA_VER alternatives: cuda90, cuda92, cuda100
ARG JAX_CUDA_VER=cuda100
RUN pip3 install --upgrade https://storage.googleapis.com/jax-wheels/${JAX_CUDA_VER}/jaxlib-0.1.8-${JAX_PY_VER}-none-linux_x86_64.whl
RUN pip3 install --upgrade jax  # install jax

# You can test tensorflow-gpu in a --runtime=nvidia session with:
# python -c "import tensorflow as tf; tf.test.is_gpu_available()"
# Test pytorch in a --runtime=nvidia session with:
# python -c "import torch; print(torch.rand(2,3).cuda())"

# add tensorflow tutorial notebooks
RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_text_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/basic_regression.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_restore_models.ipynb

# Add fastai course material to image
RUN mkdir -p /fastai/ && chmod -R a+rwx /fastai/
RUN git clone --depth=1 https://github.com/fastai/course-v3 /fastai/course-v3

# Add workshop course material to image
# Username and password are required until I make the repo public.
RUN mkdir /indl/ && chmod -R a+rwx /indl
RUN git clone https://github.com/SachsLab/IntracranialNeurophysDL.git /indl

# Link all of the notebooks into one common folder that will be easier to navigate.
RUN rm -Rf /notebooks
RUN mkdir -p /notebooks && chmod -R a+wrx /notebooks
RUN ln -s /tf/tensorflow-tutorials /notebooks/tf
RUN ln -s /fastai/course-v3/nbs /notebooks/fastai
RUN ln -s /indl/lessons /notebooks/indl

# We want to have PERSISTENT storage on the host that will survive stopping/restarting the container.
# This is mapped during image run with `-v <host/dir>:<container/dir>`
# Then we link it into the notebooks/data folder so our notebooks have easier access to it.
RUN rm -Rf /persist
RUN mkdir -p /persist && chmod -R a+wrx /persist
RUN ln -s /persist /notebooks/data
RUN ln -s /persist /root/.fastai/data

# Set the working directory for when the container is run.
WORKDIR /notebooks

# I have no idea what this does.
RUN ${PYTHON} -m ipykernel.kernelspec

# Run the jupyter notebook
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://127.0.0.1:8888'"]
