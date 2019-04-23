# Syllabus

The workshop spans 2 days.
The first half of day 1 will introduce the technologies and concepts required to be a deep learning practitioner.
The second half of the first day will use real intracranial neurophysiological data to learn about convolutional neural networks, and how they can be used in neuroscience.
Day 2 will begin with an introduction to recurrent neural networks as applied to intracortical spiking data, and then through a couple advanced architectures that make use of RNNs including LFADS and Transformers.
The second half of Day 2 is devoted to brain-inspired DL and DL-inspired neuroscience.

## Table of Contents

TODO: Fill in later

## Day 1 AM

### Part 1: Getting Started with Deep Learning Tools

60 minutes + 30 minute break

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

### Part 2: My first neural net

90 minutes

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
    * Sequential vs functional
    * tf eager
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
    * Data augmentation

## Day 1 PM

### Part 3: Introduction to CNNs

90 minutes

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
    * [e.g.](https://towardsdatascience.com/interpretable-ai-or-how-i-learned-to-stop-worrying-and-trust-ai-e61f9e8ee2c2)
* Stacking CNN with other layers - [SincNet](https://arxiv.org/abs/1808.00158)
* Sharing models across subjects

### Part 4: Variational auto-encoders

90 minutes

Dr. Guillaume Doucet will present this part.

* Introduction to dataset - Macaque PFC spiking data during ODR task.
* Process waveforms:
    * CNN architecture with middle layer with N features, outer layers to reproduce waveforms.
    * Use middle-layer activations for spike sorting
    * Use on full data as spike detection + sorting

## Day 2 AM

### Part 5: Recurrent neural nets

60 minutes

* Introduce another dataset with within-trial sequence dynamics
* Describe RNNs with pseudocode (see listing 6.21 in textbook)
* LSTM and GRU
* A few different datasets and examples are available.
    * Implementing an RNN to decode reach kinematics
        * Follow design here: [Tseng et al., 2019](https://www.ncbi.nlm.nih.gov/pubmed/30979355/)
        * Use data from joeyo or from Reach and Grasp
    * [RNN in ECoG](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6119703/)
        * Use data from Kai Miller
    * RNN for classification of MER trajectories in DBS
        * Use our trained model and one example dataset is provided

### Part 6: LFADS and Transformer

120 minutes

The RNN by itself can be quite useful, but it can also be used as a component in a larger architecture.
In this session we will briefly describe a couple different architectures that use RNNs
and test them out on some neurophys data.

* [LFADS ipynb](https://github.com/google-research/computation-thru-dynamics/blob/master/notebooks/LFADS%20Tutorial.ipynb)
* Transformer
    * [link 1](https://staff.fnwi.uva.nl/s.abnar/?p=108)
    * [Tensorflow tutorial](https://www.tensorflow.org/alpha/tutorials/text/transformer)
    * I've seen some better transformer examples. I need to search harder here.


## Day 2 PM

?? minutes

What can trained networks teach us about the brain?

* [Using trained DNNs as a model](https://www-sciencedirect-com.proxy.bib.uottawa.ca/science/article/pii/S1364661319300348)
* See Susillo training EMG generator.
* [Analyzing biological and artificial neural networks: challenges with opportunities for synergy?](https://www-sciencedirect-com.proxy.bib.uottawa.ca/science/article/pii/S0959438818301569)
    * Yamins et al PNAS 2014 compare activations in bioNN to aDNN.
* [Toward an Integration of Deep Learning and Neuroscience](https://www.frontiersin.org/articles/10.3389/fncom.2016.00094/full)

Brain-inspired deep learning?

* Differentiable neuron models
* Credit assignment

Chad notes:

[MAIN Andrew Doyle](https://brainhack101.github.io/introML/dl-course-outline.html)


