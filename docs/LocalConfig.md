## Local Configuration

* [Linux Ubuntu](#linux-ubuntu)
* [Windows 10](#windows-10)

### Linux Ubuntu

The provided instructions are intended for users working in the Ubuntu desktop environment. We will 
install nvidia-docker and run everything in a customized docker container. Advanced MacOS users
may be able to read the files in the config folder to create a local environment but such a configuration is not
supported.

1. Install `nvidia-docker` version 2.0

    Go to the [nvidia-docker Wiki](https://github.com/NVIDIA/nvidia-docker/wiki) and click on the link for
    Installation under the Version 2.0 header in the navigation bar on the right.
    ([direct link](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)))

1. Run the custom docker image

    * Try this first: `TODO: put the built image on dockerhub so it can be pulled directly`
    * Else: Build the docker image
        * If you haven't already, clone this repository locally or otherwise download the contents of the
        `config` folder then change to the path containing `indl_course.Dockerfile`.
        * `docker build -f indl_course.Dockerfile --build-arg GHUSER=<user> --build-arg GHPASS=<pass> -t indl_course .`
            * TODO: Remove the GHUSER and GHPASS after this repo goes public.
            * This takes a long time.
            * The dockerfile is based off `nvidia/cuda:9.0-base-ubuntu16.04` and mixes elements of the dockerfile
            for `tensorflow/tensorflow:latest-gpu-py3` [here](https://github.com/tensorflow/tensorflow/blob/479abd88927e54205ea418f68e64057e5b837e2d/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile)
            and the dockerfile for `paperspace/fastai:cuda9_pytorch0.3.0` [here](https://github.com/Paperspace/fastai-docker/blob/master/Dockerfile).
    * Test the image 
        * `docker run --runtime=nvidia --rm indl_course nvidia-smi`
        The expected result should be similar to below:
        ```
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 390.77                 Driver Version: 390.77                    |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |===============================+======================+======================|
        |   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
        | 23%   27C    P8     9W / 250W |    526MiB / 12188MiB |      1%      Default |
        +-------------------------------+----------------------+----------------------+
        ```
        * `docker run --runtime=nvidia --rm indl_course python -c "import tensorflow as tf; tf.test.is_gpu_available()"`
        * `docker run --runtime=nvidia --rm indl_course python -c "import torch; print(torch.rand(2,3).cuda())"`

1. Download datasets

    Optional - For fastai v2
      * `wget http://files.fast.ai/data/dogscats.zip -P ~/data/ && unzip ~/data/dogscats.zip -d ~/data/`
        
    TODO: reach and grasp

1. Run the docker container

    * Open a terminal and change to a folder that you have write access to (e.g. `cd ~`), and `mkdir data`
        * This will be your 'persistent' storage, where downloaded data and calculated models will be stored.
        If you restart your docker container you should not have to download this data again.
    * `docker run --runtime=nvidia --rm -d --name my_indl -p 8888:8888 -v $PWD/data:/root/data indl_course`
        * This will run the container in the background.
        * If you need to inspect the container then you may connect with a bash shell: `docker exec -it my_indl bash`

1. Connect to the jupyter notebook server
    * You will connect to the jupyter notebook server using your web browser. To get the URL, do one of the following:
        * `docker logs my_indl`
        * `docker exec my_indl jupyter notebook list`
    * Copy-paste the URL into your web browser and go.


### Windows 10

These instructions are provided and tested for Windows 10 but should work with previous versions, assuming users download
the proper software versions. Unlike Linux, Windows can't forward GPU drivers to a docker container so
all packages will be installed on the local machine. Note that installing outside the default directories (e.g. C:\Users\USER) might require admin privileges. 

1. Create a base Deep Learning directory that will contain all the course material (e.g. <strong> D:\DL\ </strong> )

1. Install nvidia CUDA toolkit version 9.0 for your windows version from [nvidia](https://developer.nvidia.com/cuda-90-download-archive) 
    
1. Download and install Git for Windows from [Git](https://gitforwindows.org/)

1. Download and install the latest Anaconda version from [Anaconda](https://www.anaconda.com/download/#download)

    * Launch the `Anaconda Prompt`. If Anaconda was not installed for a single user (i.e. outside of C:\Users\USER) you must run it using admin privileges by right clicking on the executable and selecting `More / Run as administrator`. 
    * Navigate to the Deep Learning directory, for example:
        * `D:`
        * `cd DL`
    * Run the following: 
        * `conda update -y -n base -c defaults conda`
        * `conda config --add channels conda-forge`
    * Create a new conda environment containing all the required packages and python version 3.6
        * `conda create -y -n indl python=3.6 pip cudatoolkit=9.0 tensorflow-gpu jupyterlab jupyter_contrib_nbextensions bottleneck matplotlib numexpr pandas packaging Pillow requests bcolz opencv seaborn python-graphviz scikit-learn ipywidgets`
    * Activate the new environment
        * `conda activate indl`
    * Add additional packages
        * `pip install sklearn-pandas pandas-summary isoweek`
    * If the installation differs from Windows, Anaconda, Python 3.6 or CUDA 9.0, go to [Pytorch.org](https://pytorch.org/get-started/locally/) and generate the appropriate command line. If not, use: 
        * `conda install pytorch torchvision -c pytorch`
    * Fast.ai course material. Replace <strong>D:\DL\ </strong> by your Deep Learning directory. 
        * `git clone --depth=1 https://github.com/fastai/fastai D:\DL\fastai`
        * `pip install D:\DL\fastai`
        * `pip install torchtext`
        * Download data [here](http://files.fast.ai/data/dogscats.zip) and unzip to D:\DL\fastai\data\dogscats
    * Fast.ai V3 material
        * `git clone https://github.com/fastai/course-v3.git D:\DL\fastai_v3`
    * Tensorflow material
        * `git clone --depth=1 https://github.com/tensorflow/docs.git D:\DL\tensorflow`
        
1. Launch Jupyter and create a new Notebook

    * `jupyter notebook`
    * Select: New / Python 3
        
1. Test installation
    * Running this line should return True: 
        * `import tensorflow as tf; tf.test.is_gpu_available()`
    * Running this line should return a 2 x 3 tensor: 
        * `import torch; print(torch.rand(2,3).cuda())`

