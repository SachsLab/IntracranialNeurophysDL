ARG UBUNTU_VERSION=18.04
ARG PY_VERSION=3.6
FROM nvidia/cuda:10.0-base-ubuntu${UBUNTU_VERSION}

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
#RUN conda config --append channels conda-forge

RUN conda create -y -n indl python=${PY_VERSION} pip cudatoolkit=10.0 tensorflow-gpu fastai\
 jupyterlab jupyter_contrib_nbextensions opencv seaborn python-graphviz scikit-learn ipywidgets\
  -c anaconda -c conda-forge -c pytorch -c fastai
ENV PATH /root/miniconda/envs/indl/bin:$PATH

RUN mkdir -p /root/.torch/models
RUN mkdir -p /root/.fastai/data

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
ARG GHUSER
ARG GHPASS
RUN mkdir /indl/ && chmod -R a+rwx /indl
RUN git clone https://${GHUSER}:${GHPASS}@github.com/SachsLab/IntracranialNeurophysDL.git /indl

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

# Set the working directory for when the container is run.
WORKDIR /notebooks

# Run the jupyter notebook
CMD ["bash", "-c", "source ~/.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://127.0.0.1:8888'"]
