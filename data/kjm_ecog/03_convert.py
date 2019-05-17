# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Chadwick Boulay
@author: Anahita Malvea
This must be run from the ../.. directory (parent/parent)
"""
import csv
from pathlib import Path
import logging
from data.utils.ImportKJM import ImportKJM
import neuropype.nodes as nn


TVLDA_SEGMENT = [-0.2, 0.6]
PSD_SEGMENT = [-0.3, 0.7]
ERP_SEGMENT = [-0.2, 0.4]
ERP_BASELINE = [-0.2, 0.05]
KEEP_F_BANDS = [1.5, 4,
                4, 7,
                8, 15,
                16, 31,
                32, 55,
                70, 130,
                130, 350]
WIN_DUR = 0.05
NTRIALS = 300


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    working_dir = Path.cwd() / 'data' / 'kjm_ecog'
    studies_file = working_dir / 'studies.csv'

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

        filename = working_dir / 'download' / row['dataset'] / 'data' / row['participant'] / row['fname']
        data_pkt = ImportKJM(filename=str(filename))()

        # Create the path where we will save the processed data.
        out_dir = working_dir / 'converted' / row['dataset']
        out_dir.mkdir(parents=True, exist_ok=True)

        # Common-average referencing.
        data_pkt = nn.Rereferencing()(data=data_pkt)

        # Simple high-pass filtering
        data_pkt = nn.IIRFilter(frequencies=[1], mode='highpass', offline_filtfilt=True)(data=data_pkt)

        # Notch filter out powerline noise. TODO: Also filter out harmonics with a Comb filter.
        data_pkt = nn.IIRFilter(frequencies=[57, 63], mode='bandstop', offline_filtfilt=True)(data=data_pkt)

        # TODO: Also get behaviour, if applicable for this dataset.
        # Note that there needs to be a way to associate the data and behaviour.
        # e.g. the behaviour probably has timestamps,
        # and the data might not but it has a sampling rate, so all we need is a time offset for the data's
        # first sample.
        # You will likely have to read more info from the csv files to find/fill in the required data.

        # Output raw segments to file.
        raw_segs = nn.ZScoring(axis='time')(data=data_pkt)
        raw_segs = nn.Segmentation(time_bounds=TVLDA_SEGMENT)(data=raw_segs)
        output_path = out_dir / (row['participant'] + '_full.h5')
        nn.ExportH5(filename=str(output_path))(data=raw_segs)

        # Process to get band powers.
        band_pows = nn.SpectrallyWhitenTimeSeries()(data=data_pkt)
        band_pows = nn.FilterBank(frequency_edges=KEEP_F_BANDS)(data=band_pows)  # Returns analytic signal.
        band_pows = nn.ShiftTimestamps(compensate_filter_lag=True)(data=band_pows)  # Undo filter delay.
        band_pows = nn.Magnitude()(data=band_pows)  # Analytic to amplitude.
        band_pows = nn.ToDecibels(source_measure='amplitude')(data=band_pows)  # Amplitude to power in decibels.

        # Bin using average of 0.05 s non-overlapping windows.
        band_pows = nn.ShiftedWindows(win_len=WIN_DUR, offset_len=WIN_DUR, unit='seconds')(data=band_pows)
        band_pows = nn.Mean(axis='time')(data=band_pows)  # Take the average value within each window.
        # Clean up windowing to get back time series, now at 20 Hz (1 / WIN_DUR).
        band_pows = nn.StripSingletonAxis(axis='time')(data=band_pows)
        bp_sig_only = nn.ExtractStreams(stream_names=['signals'])(data=band_pows)
        bp_sig_only = nn.OverrideAxis(old_axis='instance', new_axis='time')(data=bp_sig_only)
        band_pows = nn.MergeStreams(replace_if_exists=True)(data1=band_pows, data2=bp_sig_only)

        # Z-score and segment.
        band_pows = nn.ZScoring(axis='time')(data=band_pows)
        band_pows = nn.Segmentation(time_bounds=TVLDA_SEGMENT)(data=band_pows)

        output_path = out_dir / (row['participant'] + '_bp.h5')
        nn.ExportH5(filename=str(output_path))(data=band_pows)
