## Local Configuration

Before following the instructions in this document, you should follow the instructions in
the [BeforeTheWorkshop document](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/BeforeTheWorkshop.md).

Note that some of the instructions below include directions to download test data and run other test code.
This is no longer strictly necessary as we have our own tests now, but we left in the instructions for the extra curious types.

Follow the instructions for your operating system.

* [Linux Ubuntu](#linux-ubuntu)
* [Windows 10](#windows-10)
* [MacOS](#macos)

### Linux Ubuntu

The provided instructions are intended for users working in the Ubuntu 18.04 desktop environment.
The original instructions were to use nvidia-docker and they are preserved as method B.
However, the new recommended method is to setup a local environment described in method A.
Method A is easier to follow and easier to use after-the-fact, though its setup is much more likely to fail.

#### Linux Method A: Local Config

1. Identify the version of tensorflow you will be using and its requirements.
    * Look for the `tensorflow_gpu` entries in [this table](https://www.tensorflow.org/install/source#tested_build_configurations).
    * Find the latest version of tensorflow_gpu, identify the highest version of python it requires,
     and the version of CUDA it requires. As of this writing: tensorflow_gpu >= 1.13.1 with python 3.6, CUDA 10.0, and cuDNN 7.4.
1. If you have previously installed a newer version of CUDA or nvidia drivers then uninstall them.
    * `sudo apt-get --purge remove "*cublas*" "cuda*" "libcud*"`
    * `sudo apt-get --purge remove "*nvidia*"`
    * Reboot
1. Install CUDA.
    * `sudo apt-get install linux-headers-$(uname -r)`
    * Edit your `~/.profile` and add the following line:
        * `export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"`
    * Log out of your Ubuntu session and log back in.
    * In the following instructions, skip the step to Install NVIDIA driver as they will get installed with CUDA.
        * [Follow these instructions](https://www.tensorflow.org/install/gpu#install_cuda_with_apt).
        * [Alternative Link](https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu/1077063#1077063)
            * Note: Use `cuda-10-0` instead of `cuda-10-1`.
        * TensorRT is optional.
1. Make sure you have previously followed the BeforeTheWorkshop instructions.
1. Activate your `indl` conda environment.
1. Install TensorFlow: `pip install tensorflow-gpu==2.0.0-beta0`
1. Test the environment
        * `python -c "import tensorflow as tf; tf.test.is_gpu_available()"`
        * The output should be self-explanatory, except you can ignore warnings about not using CPU instructions.
1. `pip install hyperopt`
1. `pip install --upgrade https://storage.googleapis.com/jax-wheels/cuda100/jaxlib-latest-cp36-none-linux_x86_64.whl`
1. `pip install --upgrade git+https://github.com/google/jax.git`
    

#### Linux Method B: Using Docker
1. Install the nVidia driver following [these instructions](https://www.tensorflow.org/install/gpu#install_cuda_with_apt), but stop after the line that says `Reboot. Check that GPUs are visible using the command: nvidia-smi`
1. Install `nvidia-docker` version 2.0
    Go to the [nvidia-docker Wiki](https://github.com/NVIDIA/nvidia-docker/wiki) and click on the link for
    Installation under the Version 2.0 header in the navigation bar on the right.
    ([direct link](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)))
1. (Optional) Clean out old docker images
    If it has been a while since you previously configured a docker image for this workshop, then you may wish to
    cleanup your docker environment and start again.
    * List any running containers: `docker ps -a`
    * Kill any running containers: `docker stop $(docker ps -a -q)`
    * List docker images: `docker image ls`
    * Copy the IMAGE ID to clipboard
    * Remove the image: `docker image rm <pasted_image_id>`
    * Cleanup: `docker system prune`
1. Create the docker image
    * If we have a custom built image (not yet): `TODO: put the built image on dockerhub so it can be pulled directly`
    * Else: Build the docker image 
        * If you haven't already, clone this repository locally or otherwise download the `indl_workshop.Dockerfile`.
        * `docker build -f indl_workshop.Dockerfile --build-arg PYTHON=python3.6 -t indl_workshop .`
            * This takes a long time. And depending on your connection, it may stop half way and you'll have to try again.
            * The dockerfile is based off [`tensorflow/tensorflow:latest-gpu-py3-jupyter`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles)
    * Test the image 
        * `docker run --runtime=nvidia --rm indl_workshop nvidia-smi`
        The expected result should be similar to below:
        ```
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |===============================+======================+======================|
        |   0  GeForce GTX 1070    Off  | 00000000:05:00.0  On |                  N/A |
        |  0%   53C    P8    18W / 220W |    590MiB /  8116MiB |      2%      Default |
        +-------------------------------+----------------------+----------------------+
        ```
        * `docker run --runtime=nvidia --rm indl_workshop python -c "import tensorflow as tf; tf.test.is_gpu_available()"`
            * Warnings about "Your CPU supports instructions..." can be safely ignored.
        * `docker run --runtime=nvidia --rm indl_workshop python -c "import torch; print(torch.rand(2,3).cuda())"`
1. Run the docker container
    * `docker run --runtime=nvidia --rm -d --name my_indl --ipc=host -p 8888:8888 -v $PWD/indl:/persist indl_workshop`
        * This will run the container in the background.
        * If you need to inspect the container then you may connect with a bash shell: `docker exec -it my_indl bash`

1. Download datasets

    We will be downloading datasets into a folder that is available on the host computer, and that will be mounted
    in the docker container when it is run.
        * Change to a directory with lots of storage where you have write access.
        * `mkdir -p $PWD/indl/data`
        * This will be your 'persistent' storage, where downloaded data and calculated models will be stored.
        If you restart your docker container you should not have to download this data again.
    
    Optional - For fastai
      * DEPRECATED: `wget http://files.fast.ai/data/dogscats.zip -P $PWD/indl/data && unzip $PWD/indl/data/dogscats.zip -d $PWD/indl/data/`
        
    TODO: others. See [data README](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/data/README.md)

1. Connect to the jupyter notebook server
    * You will connect to the jupyter notebook server using your web browser. To get the URL, do one of the following:
        * `docker logs my_indl`
        * `docker exec my_indl jupyter notebook list`
    * Copy-paste the URL into your web browser and go.


### Windows 10

These instructions are provided and tested for Windows 10 but should work with previous versions, assuming users download
the proper software versions. Unlike Linux, Windows can't forward GPU drivers to a docker container so
all packages will be installed on the local machine.

1. Identify the version of tensorflow you will be using and its requirements.
    * Look for the `tensorflow_gpu` entries in [this table](https://www.tensorflow.org/install/source#tested_build_configurations).
    (Even though the table is for linux, the version dependencies are true in Windows too)
    * Find the latest version of tensorflow_gpu, identify the highest version of python it requires,
     and the version of CUDA it requires. As of this writing: tensorflow_gpu >= 1.13.1 with python 3.6 and CUDA 10.0

1. Install a version of nVidia drivers with version number >= to the minimum required.  
    * Go to [this table](https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html) and
determine the minimum nVidia graphics driver version required compatible with the version of CUDA identified above.
    nVidia drivers can be installed several different ways. If you already have geForce experience on your computer then
    use that. Otherwise go to nvidia.com and download drivers from there.
    
1. Next we install CUDA and add it to the PATH. There is a longer [install guide here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
or you can follow the shorter steps below.
    1. CUDA Toolkit
        * Download and run the CUDA Toolkit <version> installer [from here](https://developer.nvidia.com/cuda-toolkit-archive)
        * In the installer, choose Custom installation (Advanced), Uncheck everything,
        then check only CUDA/Development and CUDA/Runtime/Libraries.
    1. CUDNN
        * Download the CUDNN package [from here](https://developer.nvidia.com/rdp/cudnn-download)
            * Make sure the cuDNN version says "for CUDA <version>" where <version> matches the CUDA toolkit above.
        * Extract the 'cuda' folder somewhere convenient. I put it in E:\SachsLab\Tools\Misc
    1. Open your System Environment settings and add the following items to the top of the PATH
        ([general directions](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/)):
        * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin (might already be there)
        * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64
        * <path to cuddnn bin> e.g. E:\SachsLab\Tools\Misc\cuda\bin
    1. If your Anaconda Prompt is already open, close it and re-open it.
    
1. Use an Anaconda Prompt to add deep-learning related Python packages and libraries.
    * `pip install tensorflow-gpu==2.0.0-beta0`
    * Test the environment
        * `python -c "import tensorflow as tf; tf.test.is_gpu_available()"`
        * The output should be self-explanatory, except you can ignore warnings about not using CPU instructions.
    * Though not used for the workshop, install Pytorch so you can follow other Pytorch tutorials.
        * Go to [Pytorch.org](https://pytorch.org/get-started/locally/) and generate the appropriate command line.
        Be sure to select `conda` and the appropriate cudatoolkit version used above. 
        * As of this writing: `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
        * Test: `python -c "import torch; print(torch.rand(2,3).cuda())"`
        * The end of the output should read something like `device='cuda:0'`
    * `conda install hyperopt`

1. Change to the directory from which you cloned the workshop material (e.g. <strong> D:\DL\ </strong> )
    * `D:`
    * `cd DL`
        
1. Download some tutorial material
    * Fast.ai course material. Replace <strong>D:\DL\ </strong> by your Deep Learning directory. 
        * `git clone --depth=1 https://github.com/fastai/fastai D:\DL\fastai`
        * `pip install D:\DL\fastai`
        * `pip install torchtext`
        * Download data [here](http://files.fast.ai/data/dogscats.zip) and unzip to D:\DL\fastai\data\dogscats
    * Fast.ai V3 material
        * `git clone https://github.com/fastai/course-v3.git D:\DL\fastai_v3`
    * Tensorflow material
        * `git clone --depth=1 https://github.com/tensorflow/docs.git D:\DL\tensorflow`

### MacOS

These instructions provided are tested for MacOS 10.14 Mojave, but should work with earlier versions.
The MacOS environment set up here consists of the following limitations:
* The `nvidia-docker` described in the Linux instruction above cannot be used in Windows or MacOS.
* Recent Mac hardware (iMac or Macbook Pro) do not use NVIDIA GPUs, and thus CUDA cannot be used for tensorflow.
Some operations may be much slower on a Mac because it uses the CPU only.

Ensure you have completed all the <a href="https://github.com/SachsLab/IntracranialNeurophysDL/blob/master/docs/BeforeTheWorkshop.md">MacOS python setup instructions</a>. To setup the environment, execute terminal commands under each instruction wrapped in code:
1. Check Xcode CLT install:
    * `brew config`
        * You should be able to see your CLT version, here's an example output: `macOS: 10.14.4-x86_64 \ CLT: 10.2.1.0.1.1554506761 \ Xcode: 10.2.1`
    * `xcode-select --install`
1. Activate `indl` conda environment
    * `conda activate indl`
1. Install TensorFlow:
    * `pip install --upgrade pip`
    * `pip install tensorflow==2.0.0-beta1`
1. Verify installation:
    * `python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`
    * You should get something like this `Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
        * This is normal, because TensorFlow is built without CPU extensions (such as SSE4.1, SSE4.2, AVX, AVX2, FMA, etc.)
        * Until there is GPU support for MacOS, optimizing TensorFlow can be done via <a href="https://www.tensorflow.org/install/source">building from source</a>. Use at your own risk.
1. Install additional packages
    * `pip install hyperopt`
    * `pip install --upgrade jax jaxlib`
        * Alternatively, you may build from <a href="https://github.com/google/jax#building-jax-from-source">source</a> for stability and GPU support
1. We will be saving the below materials in the following working directory: `~/Developer/indl`
    * `mkdir ${HOME}/Developer && mkdir ${HOME}/Developer/indl`
1. For the workshop and your other DL projects, you may set this working directory as your workspace path in your IDE
    * `git clone https://github.com/SachsLab/IntracranialNeurophysDL.git ${HOME}/Developer/indl/IntracranialNeurophysDL`

The following sections contain supplementary material that will not be covered in the workshop.

1. Install Pytorch so you can follow other Pytorch tutorials:
    * `conda install pytorch torchvision -c pytorch`
1. Set up fastai and TensorFlow docs:
    * Clone and install fastai:
        * `git clone --depth=1 https://github.com/fastai/fastai ${HOME}/Developer/indl/fastai`
        * `pip install ${HOME}/Developer/indl/fastai torchtext`
    * The following command converts the link and saves the downloaded files in the indl directory (ignoring remote folders):
        * `wget -p --convert-links -nH -nd -P${HOME}/Developer/indl \`
        * `http://files.fast.ai/data/dogscats.zip`
    * Unzip the downloaded file:
        * `unzip ${HOME}/Developer/indl/dogscats.zip -d ${HOME}/Developer/indl/data/`
    * Clone fastai course repository:
        * `git clone https://github.com/fastai/course-v3.git ${HOME}/Developer/indl/fastai_v3`
    * Clone TensorFlow docs:
        * `git clone --depth=1 https://github.com/tensorflow/docs.git ${HOME}/Developer/indl/tensorflow`
