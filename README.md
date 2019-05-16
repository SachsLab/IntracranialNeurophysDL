The Sachs Lab has prepared a 2-day workshop on practical deep learning (DL) applied to intracranial neurophysiology.
This repository contains documentation, code, and tutorial materials for the workshop.

# Purpose

Attendees will gain familiarity with technologies commonly used in DL (e.g., keras/tensorflow on GPU,
jupyter notebooks on the cloud), to understand DL programming paradigms (e.g., batch loading data),
and to become proficient in building, training, and evaluating deep neural networks as applied to extracellular
neurophysiology data. 

Attendees will learn how to load and process electrophysiology datasets (ECoG, single-channel
deep brain microelectrode recordings, and multichannel (~192) intracortical microelectrode array datasets). After an
introduction to DL, attendees will learn how to apply several DL algorithms and architectures to these types of data,
and finally they will explore different ways of using deep learning to advance neuroscientific endeavours. Some of the
algorithms we aim to cover include convolutional neural nets (CNN) and several flavours of recurrent neural nets (RNN).

# Prerequisites

Attendees are expected to have a basic understanding of machine learning concepts, have basic familiarity with Python
syntax, and have an interest in applying deep learning to extracellular neurophysiology data.
For each topic, attendees will work through prepared examples using real data and thus are expected to bring their own
laptop and have configured their deep learning environment (instructions will be provided in the weeks before the workshop).

# Tools in scope

The workshop is opinionated in its selection of development environment, deep learning framework, and example datasets.
There are many tools to choose from, but we choose to use the TensorFlow framework (mostly) and Keras API,
and we require a GPU (tensorflow-gpu). The workshop environment requires many more packages but mostly as dependencies;
the workshop material focuses on the use of TensorFlow.

# Getting started

## Configuring your environment

The lessons assume you have access to a machine with a CUDA-enabled nVidia GPU. Click below to get
instructions for either a *local* configuration (i.e., your own desktop or laptop), or for a *remote* configuration
(i.e., from an online service like Kaggle, Google Colab or Amazon Web Services).

* [Local configuration](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/LocalConfig.md)
* [Remote configuration](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/RemoteConfig.md)

When you have finished configuring your environment, you should be able to interact with the workshop notebooks.
Additionally, your environment should also have other notebooks for your reference, including
"Introduction to TensorFlow", and the fast.ai course notebooks.

## Working with the Workshop Notebooks

The lessons comprise a series of Jupyter Notebooks, a.k.a. interactive python notebooks with file extension `.ipynb`.

From the Jupyter website:
>The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain
live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation,
numerical simulation, statistical modeling, data visualization, machine learning, and much more.

Chad's Warning:
>While Jupyter notebooks are well-suited for didactic purposes like this course, they are the wrong choice for
real-time analysis (e.g., brain-computer interface) due to limited I/O and they are the wrong choice for the
development of python packages due to the poor support for debugging imported packages. Even though
we use Jupyter notebooks here, you should strongly consider whether or not they are the right choice for your intended
application. Note: PyCharm professional >= 2019.1 (free for academics) has support for debugging jupyter notebooks so
this may be a viable workflow for some. 

## Data Sources

We have curated several datasets specifically for this workshop. Please see the
[data README](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/data/README.md) for their descriptions.

Smaller datasets will be downloaded on demand as required by each lesson.

Larger datasets will need to be downloaded in advance. TODO: Download instructions.

# Lesson Plan

## Before the workshop.

You should have already completed either the [configuration instructions](#configuring-your-environment)
and have a jupyter notebook instance running.

If you are unfamiliar with Jupyter notebooks then you should spend a few minutes going through the
fastaiv3/dl1/00_notebook_tutorial.ipynb notebook.

Afterwards, you should go through a couple of the notebooks in `tensorflow/tutorials/keras/`, starting with
`basic_classification.ipynb`.

Keen learners may be interested in going through the rest of the fast.ai courses as well.
fast.ai uses PyTorch, which we do not use in this workshops, but the deep learning principles are the same.

## Workshop Materials

Please refer to the [Syllabus](https://github.com/SachsLab/IntracranialNeurophysDL/blob/master/docs/Syllabus.md).

Each workshop _Part_ comprises one or more notebooks found in the _notebooks_ folder. Follow the notebooks in order
as outlined in the Syllabus.
