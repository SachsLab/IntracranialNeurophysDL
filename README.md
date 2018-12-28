# Introduction

This repository contains resources for the SachsLab short course on deep learning for the analysis of
intracranial extracellular electrophysiology. The course is intended for scientists and trainees who analyze
extracellular neurophysiological data (especially from human and non-human primates), have a basic
understanding of machine learning concepts, have modest programming ability, and are interested in
applying deep learning to their work. The course is scoped to get the learner applying deep learning tools
as quickly as possible, and as such the fundamentals of neural networks and deep learning are not covered.
Similarly, the course is opinionated in its selection of development environment, deep learning framework,
and example datasets. There are many tools to choose from, but we choose to use Python, with the
TensorFlow framework, and we require a GPU (tensorflow-gpu).

# Table Of Contents

* [Getting Started](#getting-started)
    * [Environment](#environment)
    * [Working with the Lesson Notebooks](#working-with-the-lesson-notebooks)
    * [Data Sources](#data-sources)
* [Lesson Plan](#lesson-plan)
    * [Lesson 1 - Working with intracranial extracellular microelectrode data in Tensorflow](#lesson-1---working-with-data-in-modern-deep-learning-platforms)
    * [Lesson 2 - Unsupervised neural networks](#lesson-2---unsupervised-neural-networks)
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
we use Jupyter notebooks here, you should strongly consider whether they are the right choice for your intended
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

Afterwards, you should go through a couple of the tensorflow/tutorials/keras/ notebooks, starting with
`basic_classification.ipynb`.

Keen learners may be interested in going through the rest of the fast.ai courses as well.
fast.ai uses PyTorch, which we do not use in this course, but the principles are the same.
Note that this image provides both fast.ai courses v2 and v3. As of this writing (December 2018),
the youtube videos are still based on the v2 courses and the v3 notebooks have some bugs.

## Lesson 1 - Working with data in modern deep learning platforms

The first lesson explores data structures in TensorFlow (tf), focusing on how to represent multichannel 
microelectrode timeseries. We will load data into tf structures, manipulate them, and visualize them.

    * Continuous vs Segmented
    * Spike data
        * Filtering and re-thresholding
        * Spike-sorting
        * Binning
        * Visualization
            * Per-unit raster plots
            * Tensor decomposition
    * Local field potentials
        * Feature extraction - is it needed?
        * Band-pass filter -> Hilbert transform
        * Connectivity
            * Phase-amplitude coupling
            
    * FP16 vs FP32

## Lesson 2 - Unsupervised neural networks

The second lesson surveys neural network techniques to learn lower-dimensional representations of data.

## Lesson 3 - 

## Lesson 4 - 