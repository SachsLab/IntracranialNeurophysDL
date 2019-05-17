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
import neuropype.engine as ne


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

        # Find Bad Time Windows. We don't remove them yet, we just identify which samples are good to keep.
        # TODO: Better parameterization.
        diag = nn.RemoveBadTimeWindows(zscore_thresholds=[-10, 12])(data=data_pkt, return_outputs='all')['diagnostic']
        b_keep = diag.chunks['signals'].block.data.astype(bool)
        bad_times = diag.chunks['signals'].block.axes[ne.time].times[~b_keep]

        # Common-average referencing.
        data_pkt = nn.Rereferencing(axis='space', reference_range=':')(data=data_pkt)

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
        raw_segs = ne.deepcopy_most(data_pkt)
        raw_segs.chunks['signals'].block = data_pkt.chunks['signals'].block[..., ne.time[b_keep], ...]
        raw_segs = nn.ZScoring(axis='time')(data=raw_segs)
        raw_segs = nn.Segmentation(time_bounds=TVLDA_SEGMENT)(data=raw_segs)
        output_path = out_dir / (row['participant'] + '_full.h5')
        nn.ExportH5(filename=str(output_path))(data=raw_segs)

        # Process data to get engineered features.

        # data_pkt = nn.SpectrallyWhitenTimeSeries()(data=data_pkt)

        # Band-pass to keep only (high gamma) broadband power
        pkt1 = nn.IIRFilter(order=8, frequencies=[50, 300], mode='bandpass', offline_filtfilt=True)(data=data_pkt)

        # Get non-overlapping 0.05 s windows.
        pkt1 = nn.ShiftedWindows(win_len=WIN_DUR, offset_len=WIN_DUR, unit='seconds')(data=pkt1)

        # Calculate variance within each window.
        pkt1 = nn.Variance(axis='time')(data=pkt1)

        # Clean up windowing to get back time series, now at 20 Hz (1 / WIN_DUR).
        pkt1 = nn.StripSingletonAxis(axis='time')(data=pkt1)
        pkt1_sig = nn.ExtractStreams(stream_names=['signals'])(data=pkt1)
        pkt1_sig = nn.OverrideAxis(old_axis='instance', new_axis='time')(data=pkt1_sig)
        # Drop bad samples from pkt1_sig
        import numpy as np
        bad_inds = np.unique(np.searchsorted(pkt1_sig.chunks['signals'].block.axes[ne.time].times, bad_times))
        b_keep = ~np.in1d(np.arange(len(pkt1_sig.chunks['signals'].block.axes[ne.time].times)), bad_inds)
        pkt1_sig.chunks['signals'].block = pkt1_sig.chunks['signals'].block[..., ne.time[b_keep], ...]
        pkt1 = nn.MergeStreams(replace_if_exists=True)(data1=pkt1, data2=pkt1_sig)
        del pkt1_sig

        # Log of variance gives an approximation of power.
        pkt1 = nn.Logarithm()(data=pkt1)
        pkt1 = nn.ZScoring(axis='time')(data=pkt1)

        # Segment data around stimulus onset.
        pkt1 = nn.Segmentation(time_bounds=TVLDA_SEGMENT)(data=pkt1)

        output_path = out_dir / (row['participant'] + '_bp.h5')
        nn.ExportH5(filename=str(output_path))(data=pkt1)

        # Just for fun, print the LDA classificationa accuracy
        lda_node = nn.LinearDiscriminantAnalysis(cond_field='Marker', shrinkage='auto')
        cv_lda = nn.Crossvalidation(method=lda_node, cond_field='Marker', folds=10)(data=pkt1, return_outputs='all')
        logging.info(" {0} - LDA Loss = {1:.3f} +/- {2:.3f}".format(
            row['participant'], cv_lda['loss'], cv_lda['loss_std']))

# Run the following command from the data/kjm_ecog folder:
# kaggle datasets version -m "Updated data." -p converted -r zip
