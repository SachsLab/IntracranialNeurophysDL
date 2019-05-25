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
laptop. Their laptop should be configured prior to the workshop (instructions will be provided in the weeks before the workshop).

# Tools in scope

The workshop is opinionated in its selection of development environment, deep learning framework, and example datasets.
There are many tools to choose from, but we choose to use the TensorFlow framework and Keras API,
and we require a GPU (tensorflow-gpu). If your laptop does not have a GPU then you can do most of the deep learning
on the cloud in Google Colab.

# Getting started

Follow the instructions in the [BeforeTheWorkshop](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/BeforeTheWorkshop.md) document.

When you have finished configuring your environment, you should be able to interact with the workshop notebooks,
either locally or remotely.
For users with GPUs, the instructions also direct you to third party notebooks for your reference, including
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

Larger datasets are not required but you may download them following the instructions in the data directory
for your own exploration outside the workshop.

# Lesson Plan

Please refer to the [Syllabus](https://github.com/SachsLab/IntracranialNeurophysDL/blob/master/docs/Syllabus.md).

Each workshop _Part_ comprises one pdf in the _slides_ folder
and one or more notebooks in the _notebooks_ folder.
Follow the slides in order and the notebooks to which they direct.
