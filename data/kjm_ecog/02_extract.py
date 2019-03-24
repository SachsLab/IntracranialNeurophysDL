# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Chadwick Boulay
@author: Anahita Malvea

This must be run from the ../.. directory (parent/parent)
Note: Sometimes the next file will start unzipping before the previous one is finished.
If the progress bar doesn't reach 100%, it might have continued to update a few lines below.
"""
import csv
import zipfile
import tqdm
from pathlib import Path


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
        print("\nExtracting {}".format(local_fname))
        with zipfile.ZipFile(file=local_fname) as _zip:
            for file in tqdm.tqdm(iterable=_zip.namelist(), total=len(_zip.namelist())):
                _zip.extract(member=file, path=local_dir)
