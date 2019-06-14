The Sachs Lab has prepared a 2-day workshop on practical deep learning (DL) applied to intracranial neurophysiology.
This repository contains documentation, code, and tutorial materials for the workshop.

# Purpose

Attendees will gain familiarity with technologies commonly used in DL (e.g., keras/tensorflow on GPU,
jupyter notebooks on the cloud), to understand DL programming paradigms, and to become proficient in building,
training, and evaluating deep neural networks as applied to extracellular neurophysiology data. 

Attendees will learn how to load and process electrophysiology datasets (ECoG, single-channel
deep brain microelectrode recordings, and multichannel (~192) intracortical microelectrode array datasets). After an
introduction to DL, attendees will learn how to apply several DL algorithms and architectures to these types of data,
and finally they will explore different ways of using deep learning to advance neuroscientific endeavours. Some of the
algorithms we cover include convolutional neural nets (CNN) and several flavours of recurrent neural nets (RNN).

# Prerequisites

Attendees are expected to have a basic understanding of machine learning concepts, have basic familiarity with Python
syntax, and have an interest in applying deep learning to extracellular neurophysiology data.
Most of the contents will apply just as well to other multi-channel timeseries data.

For each topic, attendees will work through prepared examples using real data and thus are expected to bring their own
laptop. Their laptop should be configured prior to the workshop. Instructions are below in the **Getting started** section.

# Tools in scope

The workshop is opinionated in its selection of development environment, deep learning framework, and example datasets.
There are many tools to choose from, but we choose to use the TensorFlow framework and Keras API,
and we require a GPU (tensorflow-gpu). If your laptop does not have a GPU then you can do most of the deep learning
on the cloud in Google Colab.

# Getting started

Follow the instructions in the [BeforeTheWorkshop](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/BeforeTheWorkshop.md) document.

When you have finished configuring your environment, you should be able to interact with the workshop notebooks,
either locally or remotely.

## Working with the Workshop Notebooks

The lessons comprise a series of Jupyter Notebooks, a.k.a. interactive python notebooks with file extension `.ipynb`.

From the Jupyter website:
>The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain
live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation,
numerical simulation, statistical modeling, data visualization, machine learning, and much more.

Chad's Opinion:
>While Jupyter notebooks are well-suited for didactic purposes like this course, they are the wrong choice for
real-time analysis (e.g., brain-computer interface) due to limited I/O and they are the wrong choice for the
development of python packages due to the poor support for debugging imported packages. Even though
we use Jupyter notebooks here, you should strongly consider whether or not they are the right choice for your intended
application. Note: PyCharm professional >= 2019.1 (free for academics) has support for debugging jupyter notebooks so
this may be a viable workflow for some. 

## Data Sources

We have curated several datasets specifically for this workshop. Please see the
[data README](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/data/README.md) for their descriptions.

Subsets of these datasets will be downloaded on demand as required by each lesson.

Larger datasets are not required but you may download them following the instructions in the data directory
for your own exploration outside the workshop. The data preprocessing and conversion scripts use [NeuroPype]().
NeuroPype is free for academics but the publicly available academic version is currently too old (as of June 2019).

# Lesson Plan

The Workshop comprises 8 Parts. Each Part presented by the Sachs Lab has slides and one or more Jupyter notebooks.
The slides can be found in the repository _slides_ folder (TODO: Check back here on the day of the workshop).
The notebooks can be found in the _notebooks_ folder.

Start with the slides for Part 1. The slides will then instruct the student to open one of the notebooks.
If using Google Colab, you can follow the link in the table found [in the notebooks README](https://github.com/SachsLab/IntracranialNeurophysDL/blob/master/notebooks/README.md).
