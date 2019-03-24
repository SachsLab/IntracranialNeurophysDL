# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Anahita Malvea
@author: Chadwick Boulay
"""
import os, csv, requests, zipfile, tqdm, math
import numpy as np
import scipy.io

# Get the list of studies
studies = []
with open('studies.csv') as csvfile:
    datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in datasetreader:
        studies.append(row)

# Create a local folder to store the data
local_dir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# Download the data from the server to the local folder
base_url = "https://stacks.stanford.edu/file/druid:zk881ps0522/"
for row in studies:
    fname = row['name'] + '.zip'
    remote_fname = base_url + fname
    r = requests.get(remote_fname, stream=True)  # stream=True does not download yet.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    local_fname = os.path.join(os.getcwd(), 'data', fname)
    with open(local_fname, 'wb') as f:
        for data in tqdm.tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size),
                              unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")

# Unzip each local zip file into its own directory
for row in studies:
    local_fname = os.path.join(os.getcwd(), 'data', row['name'] + '.zip')
    zip = zipfile.ZipFile(local_fname)
    zip.extractall(local_dir)
    zip.close()

# Load each .mat file, add some metadata, and save to .h5 file.
datasets = []
with open('datasets.csv') as csvfile:
    datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in datasetreader:
        datasets.append(row)

for row in datasets:
    fname = os.path.join(local_dir, row['dataset'], 'data', row['fname'])
    mat = scipy.io.loadmat(fname)
    data = mat['data']
    # TODO: Also get behaviour, if applicable for this dataset.
    # TODO: Save data, behaviour, data info (sampling rate, channel names and/or locations, impedances?)
    # into an h5 file
    # Note that there needs to be a way to associate the data and behaviour. e.g. the behaviour probably has timestamps,
    # and the data might not but it has a sampling rate, so all we need is a time offset for the data's first sample.
    # You will likely have to read more info from the csv files to find/fill in the required data.
