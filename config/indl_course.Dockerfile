
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
RUN conda update -y -n base -c defaults conda
RUN conda config --append channels conda-forge

# Create a new conda environment with many dependencies.
RUN conda create -y -n indl python=3.6 pip cudatoolkit=9.0 tensorflow-gpu jupyterlab\
 jupyter_contrib_nbextensions bottleneck matplotlib numexpr pandas packaging Pillow requests bcolz opencv\
 seaborn python-graphviz scikit-learn ipywidgets
ENV PATH /root/miniconda/envs/indl/bin:$PATH
# export PATH=/root/miniconda/envs/indl/bin:$PATH
RUN pip install sklearn-pandas pandas-summary isoweek
# Test in a --runtime=nvidia session with:
# python -c "import tensorflow as tf; tf.test.is_gpu_available()"

# Download fastai and install from its setup.py
RUN git clone --depth=1 https://github.com/fastai/fastai /root/fastai
RUN pip install /root/fastai/
RUN pip install torchtext
# Test in a --runtime=nvidia session with:
# python -c "import torch; print(torch.rand(2,3).cuda())"

# Create a notebooks dir
WORKDIR /notebooks
RUN chmod -R a+w /notebooks

# The course notebooks expect data to be stored in a specific location.
# When the docker container is run, it has mapped persistent storage. e.g. `-v $PWD/data:/root/data`
# Here we make sure that folder exists, and we link it to the expected data locations.
WORKDIR /root/data
RUN chmod -R a+w /root/data

# Download and setup learning materials from tensorflow
RUN git clone --depth=1 https://github.com/tensorflow/docs.git ~/tf_docs
RUN ln -fs ~/tf_docs/site/en /notebooks/tensorflow
WORKDIR /root/.keras
RUN ln -fs /root/data /root/.keras/datasets

# Setup fastaiv2 lessons that come with the original fastai repo we already cloned.
RUN ln -fs /root/fastai/courses /notebooks/fastaiv2
WORKDIR /root/.torch
RUN ln -fs /root/data /root/.torch/data

# Download fast.ai v3, link it into our notebooks, and link the data dir into its default dir
RUN git clone https://github.com/fastai/course-v3.git /root/fastai_v3
RUN ln -fs /root/fastai_v3/nbs /notebooks/fastaiv3
WORKDIR /root/.fastai
RUN ln -fs /root/data /root/.fastai/data

# Download the course repo so we have access to the conda environment.yml
# Username and password are required until I make the repo public.
ARG GHUSER
ARG GHPASS
RUN git clone https://${GHUSER}:${GHPASS}@github.com/SachsLab/IntracranialNeurophysDL.git ~/indl_repo
RUN ln -fs ~/indl_repo/lessons /notebooks/indl
RUN ln -fs /root/data /notebooks/indl/data

# Set the working directory for when the container is run.
WORKDIR /notebooks

# Run the jupyter notebook
EXPOSE 8888
CMD ["bash", "-c", "source ~/.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://127.0.0.1:8888'"]
