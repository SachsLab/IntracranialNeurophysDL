# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Chadwick Boulay
@author: Anahita Malvea
This must be run from the ../.. directory (parent/parent)
"""
import csv
from pathlib import Path
from data.utils.ImportKJM import ImportKJM
import neuropype.nodes as nn


TVLDA_SEGMENT = [-0.2, 0.6]
PSD_SEGMENT = [-0.3, 0.7]
ERP_SEGMENT = [-0.2, 0.4]
ERP_BASELINE = [-0.2, 0.05]
KEEP_F_BANDS = [[0, 57], [63, 117], [123, 177], [183, 201]]
WIN_DUR = 0.05
NTRIALS = 300


if __name__ == "__main__":
    working_dir = Path.cwd() / 'data' / 'kjm_ecog'
    studies_file = working_dir / 'studies.csv'
    local_dir = working_dir / 'download'

    # Get the list of studies
    studies = []
    with open(studies_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            studies.append(row)

    datasets_file = working_dir / 'datasets.csv'
    datasets = []
    with open(datasets_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            datasets.append(row)

    # Load each .mat file, add some metadata, and save to .h5 file.
    for row in datasets:
        if row['dataset'] not in ['faces_basic']:
            # For now skip datasets which are not supported.
            continue

        filename = Path(local_dir) / row['dataset'] / 'data' / row['participant'] / row['fname']
        data_pkt = ImportKJM(filename=str(filename))()
        data_pkt = nn.Rereferencing()(data=data_pkt)  # CAR
        data_pkt = nn.IIRFilter(frequencies=[1], mode='highpass', offline_filtfilt=True)(data=data_pkt)
        # Notch filter out powerline noise. TODO: Also filter out harmonics with a Comb filter.
        data_pkt = nn.IIRFilter(frequencies=[57, 63], mode='bandstop', offline_filtfilt=True)(data=data_pkt)
        # TODO: Also get behaviour, if applicable for this dataset.
        nn.ExportH5()(data=data_pkt)
        # Note that there needs to be a way to associate the data and behaviour.
        # e.g. the behaviour probably has timestamps,
        # and the data might not but it has a sampling rate, so all we need is a time offset for the data's
        # first sample.
        # You will likely have to read more info from the csv files to find/fill in the required data.
