# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Chadwick Boulay
@author: Anahita Malvea
This must be run from the ../.. directory (parent/parent)
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

    # Download the data from the server to the local folder
    base_url = "https://stacks.stanford.edu/file/druid:zk881ps0522/"
    studies_file = working_dir / 'studies.csv'
    with open(studies_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for study in datasetreader:
            fname = study['name'] + '.zip'
            remote_fname = base_url + fname
            md5 = study['md5'] if study['md5'] else None
            download_from_web(remote_fname, working_dir / 'download' / fname, md5=md5)

    # Download other files
    others_file = working_dir / 'other_files.csv'
    with open(others_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            fname = row['name']
            remote_fname = base_url + fname
            download_from_web(remote_fname, working_dir / 'download' / fname)
