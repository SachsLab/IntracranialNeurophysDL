# Preparing for the workshop

## Signing up for accounts

* You will need a laptop. You can share with a friend if you like.
    * See the [local or remote section](#local-deep-learning) below for info on laptop requirements.
* Create a GitHub account if you don't already have one.
* Setup a Google Colab account.
    * Create a google (e.g. gmail) account if you don't already have one.
    * Navigate to https://colab.research.google.com/github/ .
    * Click on "Include private repos" to authorize google colab.
* Get a kaggle account.
    * https://www.kaggle.com/
    * Download your Kaggle API key. [Instructions](https://github.com/Kaggle/kaggle-api#api-credentials)
* [Get a PyCharm student account](https://www.jetbrains.com/shop/eform/students)
    * When that comes through, download and install PyCharm professional v >= 2019.1
* [Sign up for Neuropype academic edition.](https://www.neuropype.io/academic-edition)

## Preparing a Python environment

Whether you decide to do Deep Learning locally or remotely (more on that later),
we expect you to do at least some data preprocessing locally,
so you will need Python data science tools. Don't worry if you fail to complete
this next part as we will go over it during the first workshop session (see slides
for Part1).

* Mac users [get homebrew](https://brew.sh/).
* Download and install Git
     * For Windows from [Git](https://gitforwindows.org/)
     * For Mac `brew install git`
     * For linux `sudo apt-get install git`
* Download and install the latest [miniconda](https://docs.conda.io/en/latest/miniconda.html).
    * Note that installing outside the default directories (e.g. C:\Users\<USER>) might require admin privileges.
* Update conda then create a minimal Python 3.6 environment
    * `conda update -y -n base -c defaults conda`
    * `conda config --add channels conda-forge`
    * `conda create -n indl python=3.6`
* Activate the environment
    * Windows: `conda activate indl`
    * Mac/Linux: `source activate indl`
* Install Python packages and their dependencies through conda
    * `conda install pip numpy scipy scikit-learn pandas jupyterlab jupyter_contrib_nbextensions bottleneck matplotlib numexpr packaging Pillow requests bcolz opencv seaborn python-graphviz ipywidgets tqdm watchdog qtpy cython plotly`
* Install a few additional packages through pip
    * `pip install sklearn-pandas pandas-summary isoweek kaggle`
* Clone this repository.
    * Open a terminal/anaconda prompt and cd to a directory with a lot of space. (e.g. <strong> D:\DL\ </strong> )
    * `git clone https://github.com/SachsLab/IntracranialNeurophysDL.git`
    
## Local Deep Learning

Decide if you are going to do deep learning on your local computer, or remotely.
Doing it on your local computer has some advantages (easier data management),
but the main disadvantage is that you need a decent nVidia GPU.

If your laptop has an nVidia GPU, or if you will return to a workstation with an nVidia GPU
then please continue by following the [LocalConfig](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/LocalConfig.md)
document. Otherwise you can stop here.

If  you decide to do a local config for deep learning, try running the jupyter-notebook
server and running the notebooks/01_00_tensorflow_test.ipynb notebook.
