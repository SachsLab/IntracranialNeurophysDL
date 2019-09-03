# Preparing for the workshop

* You will need a laptop. You can share with a friend if you like.
    * See the [local or remote section](#local-deep-learning) below for info on laptop requirements.
    
## Signing up for accounts

* Create a [GitHub](https://github.com/) account if you don't already have one.
* Setup a Google Colab account.
    * It works with any google account. Create a google (e.g. gmail) account if you don't already have one.
    * Authorize it to to access your github account.
        * Navigate to https://colab.research.google.com/github/ .
        * Optional: Click on "Include private repos" to authorize google colab.
* Configure your computer to download from kaggle
    * Get a kaggle account: https://www.kaggle.com/
    * Download your Kaggle API key. [Instructions](https://github.com/Kaggle/kaggle-api#api-credentials)
    * Copy your kaggle.json file into your <home>/.kaggle folder.
        * Windows users: If you have trouble creating the C:\Users\<username>\.kaggle folder,
        try executing the following commands in a Command Prompt.
        * `%systemdrive%`
        * `cd %userprofile%`
        * `mkdir .kaggle`
    * It is a good idea to also keep a copy of .kaggle in an easy-to-find directory (e.g. Downloads) 
* Optional: [Get a PyCharm student account](https://www.jetbrains.com/shop/eform/students)
    * When that comes through, download and install PyCharm professional v >= 2019.2
    * PyCharm is only required for users who wish to debug code on their own machine.
    It is highly recommended because this is a common workflow, but it is not strictly required for the workshop.
* Optional: [~~Sign up for Neuropype academic edition.~~](https://www.neuropype.io/academic-edition)
    * Neuropype is only used in a small example in the workshop itself, and if you wish to reuse
    my code to preprocess downloaded data. The workshop data are already preprocessed so it is not necessary.
    * NeuroPype academic edition works in Windows only.
    * I need to do further testing to get this working.
    
## Preparing a Python environment

Whether you decide to do _Deep Learning_ locally or remotely (more on that later),
it is common to want to do at least some _data preprocessing_ locally. Here we provide instructions on
how to configure your computer with Python-based data science tools. While this configuration is not strictly
required for the workshop, it is highly recommended that you attempt to configure your system so that you
can ask for support at the workshop should you run into any trouble during configuration.

* Mac users [get homebrew](https://brew.sh/).
* Download and install Git
     * For Windows from [Git](https://gitforwindows.org/)
     * For Mac `brew install git`
     * For linux `sudo apt-get install git`
* Download and install the latest [miniconda](https://docs.conda.io/en/latest/miniconda.html).
    * Note that installing outside the default directories (e.g. C:\Users\<USER>) might require admin privileges.
* Run the remaining commands in an Anaconda Prompt (on Windows) or a Terminal (Mac/Linux).  
* Update conda then create a minimal Python 3.6 environment
    * `conda update -y -n base -c defaults conda`
    * `conda config --add channels conda-forge`
    * `conda create -n indl python=3.6`
* Activate the environment
    * Windows: `conda activate indl`
    * Mac/Linux: `source activate indl`
* Install Python packages and their dependencies through conda
    * `conda install pip numpy scipy scikit-learn pandas jupyterlab bottleneck matplotlib numexpr packaging Pillow requests bcolz opencv seaborn python-graphviz ipywidgets tqdm watchdog qtpy cython plotly h5py jedi`
* Install a few additional packages through pip
    * `pip install sklearn-pandas pandas-summary isoweek kaggle keras-tqdm keras-vis pyreadline`
* Clone this repository and open it in PyCharm.
    * Open a terminal/anaconda prompt and cd to a directory with a lot of space. (e.g. <strong> D:\DL\ </strong> )
    * `git clone --recursive https://github.com/SachsLab/IntracranialNeurophysDL.git`
    * Run PyCharm and Open the repository root directory.
    * Configure the PyCharm IntracranialNeurophysDL project to use the indl environment.
    ([Instructions here.](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/ConfigurePyCharmCondaEnvironment.pdf))
    
## Decision: Local Deep Learning or in Cloud with Google Colaboratory? 

Decide if you are going to do deep learning on your local computer, or remotely using Google Colab.
Doing it on your local computer has some advantages (easier data management),
but the main disadvantage is that you need a decent nVidia GPU.

### Local Configuration 

If you decide to do a local configuration, either on your laptop for the workshop or on a workstation, then please
follow the [LocalConfig](https://github.com/SachsLab/IntracranialNeurophysDL/tree/master/docs/LocalConfig.md) document. 

After you've completed the local config for deep learning, try running the jupyter-notebook
server and running the notebooks/01_00_tensorflow_test.ipynb notebook.
* Open an Anaconda Prompt / Terminal
* Activate the indl environment (`conda activate indl`)
* Change to the IntracranialNeurophysDL directory (`cd /path/to/repo`)
* run `jupyter notebook`.
* In the newly launched browser, click on the notebooks folder then the notebook to launch. 

### Google Colaboratory
We suggest familiarizing yourself with colab by going over the [Welcome Notebook](https://colab.research.google.com/notebooks/welcome.ipynb) 
and the [TensorFlow GPU exercise](https://colab.research.google.com/notebooks/gpu.ipynb).
Please note that Chrome is the recommended browser.
