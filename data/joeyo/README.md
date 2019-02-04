This folder has instructions and Python code to download and convert
an open dataset from the following link:
https://zenodo.org/record/583331

You can find a publication using the dataset here:

Makin, J. G., O'Doherty, J. E., Cardoso, M. M. B. & Sabes, P. N. (2018).
Superior arm-movement decoding from cortex with a new, unsupervised-learning algorithm.
J Neural Eng 15(2): 026010. doi:10.1088/1741-2552/aa9e95

And its code here: https://github.com/jgmakin/rbmish 


The individual dataset names are listed in datasets.csv, along with their md5
checksum, and the name and md5 checksum of supplementary data if available.
Supplementary data, when present, is the broadband data recorded at
fs = 24414.0625 Hz.

The scripts expect to be located in `./data/joeyo/` relative to your working
directory.

Start with `python data/joeyo/download.py`. This will download the entire dataset
from Zenodo into the ./data/joeyo/download folder.
Note that the entire dataset is 81.3 GB and downloading it could take a while.

Then run `python data/joeyo/convert.py`. This will load each datafile, modify the
data, then save it into ./data/joeyo/converted