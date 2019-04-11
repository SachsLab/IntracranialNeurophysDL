"""
Holds some code for analyzing the faces_basic dataset.
Eventually much of this code should be broken out to functions that are common across datasets,
then this file should hold only study-specific information.
The working directory must be ../../.. relative to this file.

Notes:
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004660
    0.15 - 200 Hz 1-pole filter
    1000 Hz srate

    Paper used CAR after rejecting artifacts or epileptiform activity.
    58-62 Hz 3rd order Butterworth filter.
    400 msec stimulus on (face or house), 400 msec ISI.
    50 house and 50 face pictures per run.

    Further methods from https://www.sciencedirect.com/science/article/pii/S105381191300935X
    Spectral decoupling:
        1-sec window centred in the middle of the stimulus.
        PSD (Hann -> Fourier -> * complex conjugate)
        Normalize w.r.t. mean spectrum across all segments ( psd / mean(psd) )
        log(psd)
        PCA to get projections from PSD to PSCs (only on freqs < 200 Hz that are not around 60Hz or its harmonics)
    Online:
        Spectrogram (wavelets), project each time point onto first PSC (broadband)
        Smoothing (sigma = 0.05 sec)
        z-scoring
        exp()

Here we will take a slightly different approach:
    PSD -> TensorDecomposition (trials, frequencies, channels)
    Raw -> TensorDecomposition (trials, times, channels)
    (? DemixingPCA ?)

@author: Chadwick Boulay
"""

from pathlib import Path
import numpy as np


DATA_ROOT = Path.cwd() / 'data' / 'kjm_ecog' / 'download' / 'faces_basic'
AREA_LABELS = [
    'Temporal pole',
    'Parahippocampal gyrus',            # parahippocampal part of the medial occipito-temporal gyrus
    'Inferior temporal gyrus',
    'Middle temporal gyrus',
    'fusiform gyrus',                   # Lateral occipito-temporal gyrus,
    'Lingual gyrus',                    # lingual part of  the medial occipito-temporal gyrus
    'Inferior occipital gyrus',
    'Cuneus',
    'Post-ventral cingulate gyrus',     # Posterior-ventral part of the
    'Middle Occipital gyrus',
    'occipital pole',
    'precuneus',
    'Superior occipital gyrus',
    'Post-dorsal cingulate gyrus',      # Posterior-dorsal part of the cingulate gyrus
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    'Non-included area',
]


def import_to_npype(subject_id):
    import scipy.io
    from collections import OrderedDict
    from neuropype.engine import InstanceAxis, SpaceAxis, TimeAxis, Chunk, Block, Packet, Flags
    data_fn = DATA_ROOT / 'data' / subject_id / (subject_id + '_faceshouses.mat')
    dat_contents = scipy.io.loadmat(data_fn)
    stim = dat_contents['stim'].reshape(-1)  # samples x 1; uint8
    data = dat_contents['data']  # samples x channels; float
    srate = dat_contents['srate'][0][0]

    # Time vector
    tvec = np.arange(len(stim)) / srate

    # Process the stimulus to get an events chunk
    b_stim_onset = np.diff(np.hstack((0, stim))) != 0
    b_stim_onset = np.logical_and(b_stim_onset, stim != 0)
    stim_inds = np.where(b_stim_onset)[0]
    stim_vals = stim[stim_inds]
    stim_content = np.repeat(['ISI'], len(stim_vals)).astype(object)
    stim_content[stim_vals <= 50] = 'house'
    stim_content[np.logical_and(stim_vals > 50, stim_vals <= 100)] = 'face'
    stim_ax = InstanceAxis(tvec[b_stim_onset], data=stim_content.tolist())
    stim_ax.append_fields(['StimID'], [stim_vals])
    stim_chunk = Chunk(block=Block(data=np.nan * np.ones(stim_ax.data.shape), axes=(stim_ax,)),
                       props=[Flags.is_event_stream])

    # Get the channel labels and locations.
    locs_fn = DATA_ROOT / 'locs' / (subject_id + '_xslocs.mat')
    locs_contents = scipy.io.loadmat(locs_fn)  # 'elcode' and 'locs'
    elec_names = np.array([AREA_LABELS[el_code - 1] for el_code in locs_contents['elcode'].reshape(-1)], dtype=object)
    # Append a .N to each electrode name, where N is the count of electrodes with that name.
    # The below method is a little silly, but more straightforward approaches did not work in interactive debug mode.
    name_counts = {_: 0 for _ in np.unique(elec_names)}
    for elec_ix, elec_name in enumerate(elec_names):
        elec_names[elec_ix] = '{}.{}'.format(elec_name.title(), name_counts[elec_name])
        name_counts[elec_name] += 1

    # Put the data into a data chunk
    data_chunk = Chunk(block=Block(data=data, axes=(TimeAxis(times=tvec, nominal_rate=srate),
                                                    SpaceAxis(names=elec_names, positions=locs_contents['locs']))),
                       props=[Flags.is_signal])
    data_chunk.props['source_url'] = 'file://' + str(data_fn)

    return Packet(chunks=OrderedDict({'markers': stim_chunk, 'signals': data_chunk}))


if __name__ == "__main__":
    from neuropype.nodes import *
    import logging
    from custom import VaryingLDA
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)

    # Define segments around stimulus events.
    TVLDA_SEGMENT = [-0.2, 0.6]
    PSD_SEGMENT = [-0.3, 0.7]
    ERP_SEGMENT = [-0.2, 0.4]
    ERP_BASELINE = [-0.2, 0.05]
    KEEP_F_BANDS = [[0, 57], [63, 117], [123, 177], [183, 201]]

    # Import the data from the mat file.
    pkt = import_to_npype('de')
    pkt = Rereferencing()(data=pkt)  # CAR
    pkt = IIRFilter(frequencies=[1], mode='highpass', offline_filtfilt=True)(data=pkt)

    # TVLDA method

    # Notch filter out powerline noise. TODO: Also filter out harmonics with a Comb filter.
    pkt1 = IIRFilter(frequencies=[57, 63], mode='bandstop', offline_filtfilt=True)(data=pkt)

    # Spectrally whiten the data with AR model convolution.
    # See https://martinos.org/mne/stable/auto_examples/time_frequency/plot_temporal_whitening.html
    pkt1 = SpectrallyWhitenTimeSeries(order=10)(data=pkt1)

    # Band-pass to keep only (high gamma) broadband power
    pkt1 = IIRFilter(order=8, frequencies=[50, 300], mode='bandpass', offline_filtfilt=True)(data=pkt1)

    # Get non-overlapping 0.05 s windows.
    pkt1 = ShiftedWindows(win_len=0.05, offset_len=0.05, unit='seconds')(data=pkt1)

    # Calculate variance within each window.
    pkt1 = Variance(axis='time')(data=pkt1)

    # Clean up windowing to get back time series, now at 20 Hz (1 / 0.05).
    pkt1 = StripSingletonAxis(axis='time')(data=pkt1)
    pkt1_sig = ExtractStreams(stream_names=['signals'])(data=pkt1)
    pkt1_sig = OverrideAxis(old_axis='instance', new_axis='time')(data=pkt1_sig)
    pkt1 = MergeStreams(replace_if_exists=True)(data1=pkt1, data2=pkt1_sig)

    # Log of variance gives an approximation of power.
    pkt1 = Logarithm()(data=pkt1)
    pkt1 = ZScoring(axis='time')(data=pkt1)

    # Segment data around stimulus onset.
    pkt1 = Segmentation(time_bounds=TVLDA_SEGMENT)(data=pkt1)

    # Classify data using time-varying LDA.
    tvlda_res = VaryingLDA(independent_axis='time', cond_field='Marker',
                           n_components=5)(data=pkt1, return_outputs='all')
    MeasureLoss(cond_field='Marker')(data=tvlda_res['data'])
    tvlda_report = TensorDecompositionPlot(iv_field='Marker')(data=tvlda_res['model'], return_outputs='all')['report']

    # Visualize components using tensor decomposition
    tca_node = TensorDecomposition(num_components=20)
    tca_res = tca_node(data=pkt1, return_outputs='all')
    tca_report = TensorDecompositionPlot(iv_field='Marker')(data=tca_res['model'], return_outputs='all')['report']

    # Visualize components using dPCA
    dpca = DemixingPCA(cond_field='Marker')(data=pkt1)

    file_info = FileInfoExtraction()(data=pkt1, return_outputs='all')['report']
    ReportGeneration(report_name=str(DATA_ROOT / 'test.html'))(
        file_info=file_info,
        in0=tvlda_report,
        in1=tca_report)

    # Basic ML
    predictions = LogisticRegression(cond_field='Marker', multiclass='multinomial', max_iter=1000)(data=tca_res['data'])
    MeasureLoss(cond_field='Marker', loss_metric='MCR')(data=predictions)
