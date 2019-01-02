### Local Configuration

The provided instructions are intended for users working in the Ubuntu desktop environment. We will 
install nvidia-docker and run everything in a customized docker container. Advanced MacOS and Windows users
may be able to read the files in the config folder to create a local environment but such configurations are not
supported.

1. Install `nvidia-docker` version 2.0

    Go to the [nvidia-docker Wiki](https://github.com/NVIDIA/nvidia-docker/wiki) and click on the link for
    Installation under the Version 2.0 header in the navigation bar on the right.
    ([direct link](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0%29))

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