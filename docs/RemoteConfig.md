### Remote Configuration

This document is a bit out of date.
[Alternative cloud platforms](https://www.dataschool.io/cloud-services-for-jupyter-notebook/)

Navigate to https://colab.research.google.com/github/ .
Click on "Include private repos" to authorize google colab.


Here we describe how to use Amazon Web Services (AWS) to run the course material in the cloud.
We have created a [customized image](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html)
based on the Deep Learning Amazon Machine Image (DLAMI) with Conda described
[here](https://docs.aws.amazon.com/dlami/latest/devguide/overview-conda.html). Our customizations include:
* Installed additional packages:
   * [fast.ai](https://github.com/fastai/fastai) on the `pytorch_p36` env
   * Pillow on both `pytorch_p36` and `tensorflow_p36` envs
* Configured to automatically download data on startup.
* Notebooks for TensorFlow tutorials, fasta.ai course v2, fast.ai course v3.
* Notebooks for the SachsLab intracranial neurophysiology deep learning course.

The image will run on an AWS EC2 GPU instance 
([more info](https://docs.aws.amazon.com/dlami/latest/devguide/instance-select.html)).
The instance uses Elastic Block Store [EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AmazonEBS.html) for storage.

You will need to create an AWS account (credit card needed). Once logged in, follow
[these instructions](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/)
with a few changes:
 * Instead of using the AWS-provided DLAMI, you will use the AMI that we provide (TODO: Details).
 * Instead of using the p3.2xlarge, you can use p2.xlarge (~1/10 speed for reduction from $3/hr to $0.9/hr)
 or even a slower and cheaper non-gpu instance if you are just looking around (e.g. c5.xlarge at $0.17/hr).
 * Before you type `jupyter notebook` to launch the notebook server, activate the appropriate python environment:
    * `source activate tensorflow_p36` for TensorFlow tutorials, and this course.
    * `source activate pytorch_p36` for fast.ai tutorials.