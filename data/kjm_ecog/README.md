# Summary

This section of the repository contains code and support files to download ECoG data from Kai Miller's
repository [(here)](https://exhibits.stanford.edu/data/catalog/zk881ps0522),
to load it into Python, and to save it in a common intermediate format.

The list of studies to be processed is in studies.csv,
The list of datasets for each study is in datasets.csv.
The list of participants, including their electrodes, is in participants.csv.
TODO: participants.csv is incomplete.

# Description

# Download and Convert

The scripts expect to be located in `./data/kjm_ecog/` relative to your working
directory.

Start with `python data/kjm_ecog/01_download.py`.
This will download each study's zip file into the ./data/kjm_ecog/download folder.
Note that the entire dataset is 7.6 GB and downloading it could take a while.
This server can sometimes be quite slow when downloading with this script.
It might be faster to download the data using your browser. 

Then run `python data/kjm_ecog/02_extract.py`. This will unzip each file to its own folder.

(TODO:) Then run `python data/kjm_ecog/03_convert.py`. This will perform preliminary processing
 on the data and store the result in ./data/kjm_ecog/convert
