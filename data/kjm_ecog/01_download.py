# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Chadwick Boulay
@author: Anahita Malvea
"""
import csv
from pathlib import Path
from data.utils import download_from_web


if __name__ == "__main__":
    working_dir = Path.cwd() / 'data' / 'kjm_ecog'

    # Create a local folder to store the data
    local_dir = working_dir / 'download'
    if not local_dir.is_dir():
        local_dir.mkdir()

    # Get the list of studies
    studies_file = working_dir / 'studies.csv'
    studies = []
    with open(studies_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            studies.append(row)

    # Download the data from the server to the local folder
    base_url = "https://stacks.stanford.edu/file/druid:zk881ps0522/"
    for row in studies:
        fname = row['name'] + '.zip'
        remote_fname = base_url + fname
        md5 = row['md5'] if row['md5'] else None
        download_from_web(remote_fname, working_dir / 'download' / fname, md5=md5)

    # Download other files
    others_file = working_dir / 'other_files.csv'
    with open(others_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            fname = row['name'] + '.zip'
            remote_fname = base_url + fname
            download_from_web(remote_fname, working_dir / 'download' / fname)
