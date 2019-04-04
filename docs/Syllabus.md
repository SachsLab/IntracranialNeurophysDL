# Syllabus

## Table of Contents

Fill in later

## Day 1 AM Part 1: Getting Started with Deep Learning Tools

The first session will be devoted to introducing deep learning tools, and making sure participants can get the
most out of their tools. It is expected that participants will have already received and followed provided
instructions to setup their local environment prior to attending the workshop.

* Our chosen deep learning library -- tensorflow (GPU-enabled variant) -- has requirements:
    * Match your version of tensorflow-gpu's dependencies and your hardware
    [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html) 
* Python environments
    * Understanding the PATH
    * Environment and package managers: conda, pip
        * [more info](https://medium.com/@rgalbo/simple-python-environments-for-data-science-globe-with-meridians-2b952a3f497f)
* Setting up a local environment
    * Mac/Windows/Linux: Using miniconda
* Setting up a remote environment 
    * Local example in Linux only with nvidia-docker
        * Local needs correct nvidia drivers
    * Free cloud options [described here](https://www.dataschool.io/cloud-services-for-jupyter-notebook/)
    (but ignore jupyter context for now)
* Using an IDE
    * PyCharm Professional (v>=2019)), free for academics
        * Other options not discussed: Spyder, VS Code
    * Configuring its interpreter / environment
    * How to debug
* Using Jupyter notebooks
    * Run the server (remote or local) and connect with your browser
    * Tips for working with jupyter notebooks
    * Attach to the server process with your IDE
    
During the first break we will help participants who had trouble configuring their laptop
prior to the workshop.

## Day 1 AM Part 2: My first neural net

The second session will introduce neural nets. 

* Introduce our first data set - an ECoG dataset from [Kai Miller's repository](https://exhibits.stanford.edu/data/catalog/zk881ps0522).
* Feature engineering typically required for shallow ML: expert signal processing and feature extraction
* Separate trials (where each trial has a feature vector and label) into training, validation, and test sets.
* Get common-sense baseline
    * What's the best we can do without data? Depends on number of classes and balance.
    * Try with shallow ML: LDA
        * Visualize results including weight projections on electrode grid.
    * Try with shallow ML: Varying LDA
* Introduce Keras
* Introduce tf.data
* Try a not-so-deep neural net    
    * Build a simple 1-layer Dense network with linear activations
    * Visualize with TensorBoard 
    * How is this any better than LDA? [It's not](https://www.jstor.org/stable/2584434)
* A few simple modifications
    * More layers
        * Keep adding until we overfit
    * Different activation functions
    * Regularization

## Day 1 PM Part 1: Introduction to CNNs

In the previous session we classified our highly-engineered features using a simple neural net.
This was a trivial task for the neural net to perform because we already did all of the hard
work (feature engineering) thanks to decades of research in the field.

Where deep learning really shines is in the application of generic algorithms/architectures and learning
of abstract relationships that do not rely on the reduced features. In this session we
will classify our data using one such generic architecture: a convolutional neural net. We
will also see how parts of this architecture can be shared across datasets.

* Introduction to dataset - Macaque PFC spiking data during ODR task.
* Process waveforms:
    * CNN architecture with middle layer with N features, outer layers to reproduce waveforms.
    * Use middle-layer activations for spike sorting
    * Use on full data as spike detection + sorting
* TODO: other hyper-parameters. Learning rate schedules. Weight initialization.

## Day 1 PM Part 2: Variational auto-encoders

## Day 2 AM Part 1: Recurrent neural nets

## Day 2 AM Part 2: LFADS and Transformer

## Day 2 PM

What can trained networks teach us about the brain? See Susillo training EMG generator.
Brain-inspired deep learning.

