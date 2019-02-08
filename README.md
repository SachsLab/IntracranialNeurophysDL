# Purpose

The SachsLab is preparing a short (~2-day) workshop on practical deep learning (DL) applied to intracranial neurophysiology. The goal of the workshop is to help attendees gain familiarity with technologies commonly used in DL (e.g., tensorflow on GPU, jupyter notebooks), to understand DL programming paradigms (e.g., batch loading data), and to become proficient in the application of DL to intracranial neurophysiology. The workshop is intended for scientists and trainees who have a basic understanding of machine learning concepts, have basic familiarity with Python syntax, and are interested in applying deep learning to extracellular electrophysiology data. For learners who do not have an interest in these kinds of data but are interested in DL more generally, please feel free to reach out to Chad and he will be happy to direct you to some wonderful resources that are better suited to your interests.

In the workshop, attendees will learn how to run and interact with keras/tensorflow on a GPU either locally or on a remote server. They will learn how to load and process electrophysiology datasets (1 open ECoG dataset, 1 single-channel deep brain microelectrode dataset, and 3 multichannel (~192) intracortical microelectrode array datasets). After an introduction to DL, attendees will learn how to apply several DL algorithms and architectures to these types of data, and finally they will explore different ways of using deep learning to advance neuroscientific endeavours. Some of the algorithms we aim to cover include convolutional neural nets (CNN), several flavours of recurrent neural nets (RNN), autoencoders, and transformer models. For each topic, attendees will work through prepared examples using real data and thus are expected to bring their own laptop and have configured their deep learning environment (instructions will be provided in the week before the workshop).

# Introduction

This repository contains resources for the SachsLab workshop on deep learning for the analysis of intracranial extracellular electrophysiology. The workshop is opinionated in its selection of development environment, deep learning framework, and example datasets. There are many tools to choose from, but we choose to use Python, with the TensorFlow framework, and we require a GPU (tensorflow-gpu).

# Table Of Contents

* [Getting Started](#getting-started)
    * [Environment](#environment)
    * [Working with the Lesson Notebooks](#working-with-the-lesson-notebooks)
    * [Data Sources](#data-sources)
* [Lesson Plan](#lesson-plan)
    * [Lesson 1 - Working with intracranial extracellular microelectrode data in Tensorflow](#lesson-1---working-with-data-in-modern-deep-learning-platforms)
    * [Lesson 2 - Neural networks for classification](#lesson-2---neural-networks-for-classification)
https://github.com/SachsLab/IntracranialNeurophysDL
# Getting Started

## Environment

The lessons assume you have access to a machine with a CUDA-enabled nVidia GPU. Here we provide
instructions for either a *local* configuration (i.e., your own desktop or laptop) but only for Ubuntu Linux.
We also provide instructions for a *remote* configuration (i.e., from a paid service like Amazon Web Services
or Paperspace) that applies to any computer with a web browser and keyboard.

* [Local configuration](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/LocalConfig.md)
* [Remote configuration](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/RemoteConfig.md)

Advanced MacOS or Windows users may be able to use the information in the local config docs to setup their own
local configuration but this will not be explained in this course.

We use CUDA 9.0 because its nVidia driver requirements are more easily achieved than the requirements of
CUDA 9.2 or 10.0. We use Jupyter notebooks running on Python 3.6 with [PyTorch](https://pytorch.org/) (including
[torchvision](https://pytorch.org/docs/stable/torchvision/index.html)), and the [TensorFlow](https://www.tensorflow.org/)
gpu-enabled variant: `tensorflow-gpu`.

The development environment comes with 3 sets of notebooks:
  * Introduction to TensorFlow
  * The [fast.ai](https://www.fast.ai/) course (v3).
  * The lessons for this course.

## Working with the Lesson Notebooks

The lessons comprise a series of Jupyter Notebooks, a.k.a. interactive python notebooks with file extension `.ipynb`.

From the Jupyter website:
>The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain
live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation,
numerical simulation, statistical modeling, data visualization, machine learning, and much more.

Chad's Warning:
>While Jupyter notebooks are well-suited for didactic purposes like this course, they are the wrong choice for
real-time analysis (e.g., brain-computer interface) due to limited I/O and they are the wrong choice for the
development of python packages due to the poor (non-existent) support for debugging imported packages. Even though
we use Jupyter notebooks here, you should strongly consider whether or not they are the right choice for your intended
application. 

## Data Sources

TODO

# Lesson Plan

## Before the course

You should have already completed either the [Local Configuration](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/LocalConfig.md)
or [Remote Configuration](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/RemoteConfig.md)
and have a jupyter notebook instance running.

If you are unfamiliar with Jupyter notebooks then you should spend a few minutes going through the
fastaiv3/dl1/00_notebook_tutorial.ipynb notebook.

Afterwards, you should go through a couple of the notebooks in `tensorflow/tutorials/keras/`, starting with
`basic_classification.ipynb`.

Keen learners may be interested in going through the rest of the fast.ai courses as well.
fast.ai uses PyTorch, which we do not use in this course, but the principles are the same.
Note that this image provides both fast.ai courses v2 and v3. As of this writing (December 2018),
the youtube videos are still based on the v2 courses and the v3 notebooks have some bugs.

## Lesson 1 - Working with data in modern deep learning platforms

The first lesson explores data structures in TensorFlow (tf), focusing on how to represent multichannel 
microelectrode timeseries. We will load data into tf structures, manipulate them, and visualize them.


All of the following will be done in the notebooks.

    * Other notebook tips
        * Tab-completion
        * shift+tab completion
        * ?
        * ??
        * Press "h" for keyboard shortcuts
    * Import raw data using python-neo
    * Arrays: matrices and tensors. For each of our example datasets:
        * Explain the experiment if applicable. Describe the recording setup (electrodes, amps, other measures). 
        * Print data shape
        * Print some of the contents
            * Look at scale, precision. Data should be standardized.
            * FP16 vs FP32 on GPU.
        * Print additional structure (labels, kinematics, etc.)
        * Visualize individual trials, colour coded by condition
        * Visualize condition-average (much information lost)
        * Visualize covariance structure.
        * Tensor decomposition
    * Domain expertise and feature engineering
        * Become experts in neurophysiology of PD --> beta burst length and PAC
            * BG-thalamocortical network has oscillatory activity --> time-frequency transform to spectrogram
            * Beta band-pass filter --> Hilbert transform --> Instantaneous amplitude/phase
        * Become experts in intracortical array neurophysiology --> "Neural modes"
            * High-pass filter
            * Threshold
            * Spike-sorting
            * Demultiplexing?
            * Binned spike counts
            * Counts to rates
            * Dimensionality reduction (tensor decomp; factor analysis)
        * Features can then be used in 'simpler' ML algorithm
            * Describe Linear Discriminant Analysis using engineered features
                * Analytical solution
            * Show e.g. LDA in neural network parlance. (https://www.jstor.org/stable/2584434)
                * Loss function
                * Loss gradient
                    * Learning rate
                * Why log(p) instead of accuracy?
        * In some cases, neural networks eliminate much of the need for feature engineering.
            * Indeed, with enough data, and enough parameters, it is provable that feature engineering is unnecessary.
    
## Lesson 2 - Neural networks for classification

The second lesson introduces the general idea of neural networks, building from the simplest 1-layer linear network
in the previous lesson up to deep recurrent networks. We will interact with real neurophysiological data throughout.

    * Other activation functions
        * Logistic
        * RELU
    * Calculating gradient
    * Stacking layers
    * ...

## Lesson 3 - Neural networks for feature extraction

## Lesson 4 - 

## Notes

Explained convolution kernels - http://setosa.io/ev/image-kernels
Example of non-linear function - http://neuralnetworksanddeeplearning.com/chap4.html
sigmoid vs relu
Define params vs hyper-params
Choosing hyper parameters
    Learning rate - cyclic
Loss function. Loss gradient.
