# Syllabus

The workshop spans 2 days.

The first half of day 1 will introduce the technologies and
concepts required to be a deep learning practitioner. The second half of the first day
will use real intracranial neurophysiological data to learn about convolutional neural
networks, and how they can be used in neuroscience.

Day 2 will begin with an introduction to recurrent neural networks.

## Table of Contents

Fill in later

## Day 1 AM Part 1: Getting Started with Deep Learning Tools

The first session will be devoted to introducing deep learning tools, and making sure participants can get the
most out of their tools. It is expected that participants will have already received and followed provided
instructions to setup their local environment prior to attending the workshop. Given that, we will go
quickly through this section. At the break, we will provide assistance to attendees who
had trouble setting up their environment.

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

## Day 1 AM Part 2: My first neural net

The second session will introduce neural nets and their basic components. 

* Introduce our first example data set - an ECoG dataset from [Kai Miller's repository](https://exhibits.stanford.edu/data/catalog/zk881ps0522).
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
    * How to use training loss vs validation loss.
* What is happening?
    * Gradient descent
    * Momentum, RMSProp, Adam
    * Further learning rate adaptation / schedules
* How is this any better than LDA? Naively, [it's not](https://www.jstor.org/stable/2584434)
* A few simple modifications beyond LDA
    * More layers
        * Keep adding until we overfit
    * Different activation functions
        * tanh, ReLU
    * Regularization
        * dropout

## Day 1 PM Part 1: Introduction to CNNs

In the previous session we classified our highly-engineered features using a simple neural net.
This was a trivial task for the neural net to perform because we already did all of the hard
work (feature engineering) thanks to decades of research in the field.

Where deep learning really shines is in the application of generic algorithms/architectures and learning
of abstract relationships that do not rely on the reduced features. In this session we
will classify our data using one such generic architecture: a convolutional neural net. We
will also see how parts of this architecture can be shared across datasets.

* Convolution operation
    * Translation invariance
* Convolution as a layer
* Deep network with convolutional layers
* Other tweaks:
    * Dealing with vanishing or exploding gradients with [weight initialization](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)
* Peeking inside the black box
    * 


## Day 1 PM Part 2: Variational auto-encoders

* Introduction to dataset - Macaque PFC spiking data during ODR task.
* Process waveforms:
    * CNN architecture with middle layer with N features, outer layers to reproduce waveforms.
    * Use middle-layer activations for spike sorting
    * Use on full data as spike detection + sorting

## Day 2 AM Part 1: Recurrent neural nets

## Day 2 AM Part 2: LFADS and Transformer

* [LFADS ipynb](https://github.com/google-research/computation-thru-dynamics/blob/master/notebooks/LFADS%20Tutorial.ipynb)
* [Transformer](https://staff.fnwi.uva.nl/s.abnar/?p=108)


## Day 2 PM

What can trained networks teach us about the brain?

* [Using trained DNNs as a model](https://www-sciencedirect-com.proxy.bib.uottawa.ca/science/article/pii/S1364661319300348)
* See Susillo training EMG generator.
* [Analyzing biological and artificial neural networks: challenges with opportunities for synergy?](https://www-sciencedirect-com.proxy.bib.uottawa.ca/science/article/pii/S0959438818301569)

Brain-inspired deep learning?

* Differentiable neuron models
* Credit assignment

Chad notes:

[MAIN Andrew Doyle](https://brainhack101.github.io/introML/dl-course-outline.html)

