# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Anahita Malvea
@author: Chadwick Boulay
"""
import csv, zipfile
import numpy as np
import scipy.io
from pathlib import Path
from data.utils import get_md5


if __name__ == "__main__":
    working_dir = Path.cwd() / 'data' / 'kjm_ecog'
    studies_file = working_dir / 'studies.csv'

    # Get the list of studies
    studies = []
    with open(studies_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            studies.append(row)

    # Unzip each local zip file into its own directory
    local_dir = working_dir / 'download'
    for row in studies:
        local_fname = local_dir / (row['name'] + '.zip')
        if not local_fname.exists():
            print("File {} does not exist. Skipping.".format(local_fname))
            continue
        # print("fname: {}, md5: {}".format(local_fname, get_md5(local_fname)))
        zip = zipfile.ZipFile(local_fname)
        zip.extractall(local_dir)
        zip.close()

    datasets_file = working_dir / 'datasets.csv'
    datasets = []
    with open(datasets_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            datasets.append(row)

    # Load each .mat file, add some metadata, and save to .h5 file.
    for row in datasets:
        fname = local_dir / row['dataset'] / 'data' / row['fname']
        mat = scipy.io.loadmat(fname)
        data = mat['data']
        # TODO: Also get behaviour, if applicable for this dataset.
        # TODO: Save data, behaviour, data info (sampling rate, channel names and/or locations, impedances?)
        # into an h5 file
        # Note that there needs to be a way to associate the data and behaviour. e.g. the behaviour probably has timestamps,
        # and the data might not but it has a sampling rate, so all we need is a time offset for the data's first sample.
        # You will likely have to read more info from the csv files to find/fill in the required data.
