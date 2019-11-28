# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 21:07:28 2019

@author: Chadwick Boulay
@author: Anahita Malvea
This must be run from the ../.. directory (parent/parent)
"""
from collections import OrderedDict
import csv
from pathlib import Path
import logging
import numpy as np
import neuropype.nodes as nn
import neuropype.engine as ne
from data.utils.ImportKJMFacesHouses import ImportKJMFacesHouses
from data.utils.ImportKJMFingerFlex import ImportKJMFingerFlex


SEGMENTS = {
    'faces_basic': [-0.2, 0.6],
    'fingerflex': [-0.75, 0.75]
}
KEEP_F_BANDS = OrderedDict({
    # 'Delta': [1.5, 4],
    # 'Theta': [4, 8],
    'Alpha': [8, 14],
    'Beta': [14, 32],
    # 'Gamma1': [32, 55],
    # 'Gamma2': [70, 150],
    'Broadband': [50, 300]})
WIN_DUR = 0.05  # Window for variance when calculating bandpass power.


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

    # Load each .mat file, add some metadata, do preprocessing and save to .h5 files.
    for row in datasets:
        if row['dataset'] not in ['faces_basic']:  # 'faces_basic', 'fingerflex'
            # For now skip datasets which are not supported.
            continue

        # Create the path where we will save the data.
        out_dir = working_dir / 'converted' / row['dataset']
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = working_dir / 'download' / row['dataset'] / 'data' / row['participant'] / row['fname']
        if row['dataset'] == 'faces_basic':
            sig_pkt = ImportKJMFacesHouses(filename=str(filename))()
        elif row['dataset'] == 'fingerflex':
            data_pkt = ImportKJMFingerFlex(filename=str(filename))()
            # I can't see myself ever wanting the Transition events.
            data_pkt = nn.SelectInstances(selection=[{'name': 'Marker', 'value': 'Transition'}],
                                          invert_selection=True)(data=data_pkt)
            behav_pkt = nn.ExtractStreams(stream_names=['behav'])(data=data_pkt)
            sig_pkt = nn.ExtractStreams(stream_names=['signals', 'markers'])(data=data_pkt)

            # Load the brain mesh
            import scipy.io
            brain = scipy.io.loadmat(filename, squeeze_me=True)['brain']
            verts = brain['vert'][()]
            tris = brain['tri'][()]
            brain_output_path = out_dir / (row['participant'] + '_brain.npz')
            np.savez(brain_output_path, verts=verts, tris=tris)

        sig_pkt = nn.BadChannelRemoval(corr_threshold=0.5, noise_threshold=6.,
                                       max_broken_time=0.7, use_clean_window=True)(data=sig_pkt)

        # Find Bad Time Windows. We don't remove them yet. For now we just identify which samples are good to keep.
        # TODO: Better parameterization.
        diag = nn.RemoveBadTimeWindows(zscore_thresholds=[-10, 12])(data=sig_pkt, return_outputs='all')['diagnostic']
        b_keep = diag.chunks['signals'].block.data.astype(bool)
        bad_times = diag.chunks['signals'].block.axes[ne.time].times[~b_keep]

        # Common-average referencing.
        sig_pkt = nn.Rereferencing(axis='space', reference_range=':')(data=sig_pkt)

        # Simple high-pass filtering. Probably unnecessary.
        # sig_pkt = nn.IIRFilter(frequencies=[1], mode='highpass', offline_filtfilt=True)(data=sig_pkt)

        # Notch filter out powerline noise. TODO: Also filter out harmonics with a Comb filter.
        sig_pkt = nn.IIRFilter(frequencies=[57, 63], mode='bandstop', offline_filtfilt=True)(data=sig_pkt)

        # z-score all samples, but mean and offset come from good samples.
        # zscoring should be done after import, if needed.
        if False:
            zscore_node = nn.ZScoring(axis='time')
            sig_no_outliers = ne.deepcopy_most(sig_pkt)
            sig_no_outliers.chunks['signals'].block = sig_no_outliers.chunks['signals'].block[..., ne.time[b_keep], ...]
            _ = zscore_node(sig_no_outliers)  # First to calculate mean and SD
            z_pkt = zscore_node(sig_pkt)      # Second on all data.
            del sig_no_outliers

        # Output 1: Full unsegmented data
        if row['dataset'] == 'faces_basic':
            full_pkt = sig_pkt
        elif row['dataset'] == 'fingerflex':
            full_pkt = nn.MergeStreams(replace_if_exists=True)(data1=sig_pkt, data2=behav_pkt)
        output_path = out_dir / (row['participant'] + '_full.h5')
        nn.ExportH5(filename=str(output_path))(data=full_pkt)

        # Output 2: Segmented data, timelocked to stimulus/movement onset
        # First drop 'bad' timestamps from all signal/behav chunks.
        for n in full_pkt.chunks:
            if full_pkt.chunks[n].props[ne.Flags.is_signal]:
                full_pkt.chunks[n].block = full_pkt.chunks[n].block[..., ne.time[b_keep], ...]
        if row['dataset'] == 'fingerflex':
            seg_pkt = nn.SelectInstances(selection=[{'name': 'MarkerType', 'value': 'Stim'}])(data=full_pkt)
        else:
            seg_pkt = full_pkt
        seg_pkt = nn.Segmentation(time_bounds=SEGMENTS[row['dataset']])(data=seg_pkt)
        output_path = out_dir / (row['participant'] + '_segs.h5')
        nn.ExportH5(filename=str(output_path))(data=seg_pkt)
        del seg_pkt

        # Option 3: Segmented Bandpowers
        # Process (unsegmented) data, then segment, to get engineered features.
        pow_pkt = nn.ExtractStreams(stream_names=['signals', 'markers'])(data=full_pkt)
        pow_pkt = nn.SpectrallyWhitenTimeSeries(order=10)(data=pow_pkt)
        # Band-pass in different frequency bands.
        # Note: I used 6th order butter IIRFilter because that's what was used in the TVLDA paper.
        # It might have been OK to use FIRFilterBank + timeshift correction.
        catdat_kwargs = {}
        f_centers = []
        for fp_ix, freq_pair in enumerate(KEEP_F_BANDS.values()):
            catdat_kwargs['data' + str(fp_ix + 1)] = nn.IIRFilter(frequencies=freq_pair, mode='bandpass',
                                                                  order=6, design='butter',
                                                                  offline_filtfilt=True)(data=pow_pkt)
            f_centers.append(np.mean(freq_pair))
        pow_pkt = nn.ConcatInputs(axis='frequency', create_new=True, properties=f_centers)(**catdat_kwargs)
        # Get non-overlapping 0.05 s windows.
        pow_pkt = nn.ShiftedWindows(win_len=WIN_DUR, offset_len=WIN_DUR, unit='seconds',
                                    remove_gaps=True, max_gap_length=0.0015,
                                    timestamp_origin='first')(data=pow_pkt)
        # Calculate variance within each window.
        pow_pkt = nn.Variance(axis='time')(data=pow_pkt)
        # Clean up windowing to get back time series, now at 20 Hz (1 / WIN_DUR).
        pow_pkt = nn.StripSingletonAxis(axis='time')(data=pow_pkt)
        pow_pkt = nn.OverrideAxis(old_axis='instance', new_axis='time', only_signals=True)(data=pow_pkt)
        pow_pkt = nn.OverrideSamplingRate(sampling_rate=1/WIN_DUR)(data=pow_pkt)
        # Log of variance gives an approximation of power.
        pow_pkt = nn.Logarithm()(data=pow_pkt)
        pow_pkt = nn.ZScoring(axis='time')(data=pow_pkt)

        if row['dataset'] == 'fingerflex':
            # Need to smooth and downsample behaviour data at the same rate.
            behav_pkt = nn.ExtractStreams(stream_names=['behav'])(data=full_pkt)
            behav_pkt = nn.ShiftedWindows(win_len=WIN_DUR, offset_len=WIN_DUR, unit='seconds',
                                          remove_gaps=True, max_gap_length=0.0015,
                                          timestamp_origin='first')(data=behav_pkt)
            behav_pkt = nn.Mean(axis='time')(data=behav_pkt)
            behav_pkt = nn.StripSingletonAxis(axis='time')(data=behav_pkt)
            behav_pkt = nn.OverrideAxis(old_axis='instance', new_axis='time', only_signals=True)(data=behav_pkt)
            behav_pkt = nn.OverrideSamplingRate(sampling_rate=1/WIN_DUR)(data=behav_pkt)
            pow_pkt = nn.MergeStreams(replace_if_exists=True)(data1=pow_pkt, data2=behav_pkt)
            pow_pkt = nn.SelectInstances(selection=[{'name': 'MarkerType', 'value': 'Stim'}])(data=pow_pkt)

        # Segment data around stimulus onset.
        pow_pkt = nn.Segmentation(time_bounds=SEGMENTS[row['dataset']])(data=pow_pkt)

        # Note: this output is suitable for testing different dimensionality reduction and ML approaches,
        # except when the dimensionality reduction / ML are dependent on the signal processing.
        # e.g., FBCSP, Deep Learning
        output_path = out_dir / (row['participant'] + '_bp.h5')
        nn.ExportH5(filename=str(output_path))(data=pow_pkt)

        if False:
            # Just for fun, print the LDA classification accuracy
            pow_pkt = nn.ExtractStreams(stream_names=['signals', 'markers'])(data=pow_pkt)
            pow_pkt = nn.SelectRange(axis='frequency', selection=-1)(data=pow_pkt)
            lda_node = nn.LinearDiscriminantAnalysis(cond_field='Marker', shrinkage='auto')
            cv_lda = nn.Crossvalidation(method=lda_node, cond_field='Marker', folds=10,
                                        stratified=True)(data=pow_pkt, return_outputs='all')
            logging.info(" {0} - LDA Loss = {1:.3f} +/- {2:.3f}".format(
                row['participant'], cv_lda['loss'], cv_lda['loss_std']))

# Run the following command from the data/kjm_ecog folder:
# kaggle datasets version -m "Updated data." -p converted -r zip
