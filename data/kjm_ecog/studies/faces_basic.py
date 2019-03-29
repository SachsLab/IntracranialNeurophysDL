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
    elec_names = np.array([AREA_LABELS[el_code - 1] for el_code in locs_contents['elcode'].reshape(-1)])
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
    import matplotlib.pyplot as plt


    logging.basicConfig(level=logging.DEBUG)

    # Define segments around stimulus events.
    PSD_SEGMENT = [-0.3, 0.7]
    ERP_SEGMENT = [-0.2, 0.4]
    ERP_BASELINE = [-0.2, 0.05]
    KEEP_F_BANDS = [[0, 57], [63, 117], [123, 177], [183, 201]]

    # Import the data from the mat file.
    pkt = import_to_npype('de')
    pkt = Rereferencing()(data=pkt)  # CAR

    # Get broadband power
    bb = Segmentation(time_bounds=PSD_SEGMENT)(data=pkt)
    bb = WelchSpectrum(segment_samples=0.5, unit='seconds')(data=bb)
    bb = StripSingletonAxis(axis='time')(data=bb)
    bb = Divide()(data1=bb,
                  data2=Mean(axis='instance')(data=bb))
    bb = Logarithm()(data=bb)
    # In KJM method, here is where he does PCA excluding 60 Hz harmonics and > 200 Hz
    # We will similarly eliminate these frequencies, but then do TensorDecompositionAnalysis
    fvec = bb.chunks['signals'].block.axes['frequency'].frequencies
    b_keep = np.zeros_like(fvec).astype(bool)
    for f_band in KEEP_F_BANDS:
        b_keep = np.logical_or(b_keep, np.logical_and(f_band[0] < fvec, fvec < f_band[1]))
    keep_inds = np.where(b_keep)[0]
    bb = SelectRange(axis='frequency', selection=keep_inds)(data=bb)
    tca_node = TensorDecomposition(num_components=10)
    tca_res = tca_node(data=bb, return_outputs='all')

    # TODO: Debug this plotting node, also add option to output to notebook.
    tca_report = TensorDecompositionPlot()(data=tca_res['model'], return_outputs='all')['report']
    file_info = FileInfoExtraction()(data=bb, return_outputs='all')['report']
    ReportGeneration(report_name=str(DATA_ROOT / 'test.html'))(
        file_info=file_info,
        in0=tca_report)

    # Basic ML
    predictions = LogisticRegression(cond_field='Marker', multiclass='multinomial', max_iter=1000)(data=tca_res['data'])
    MeasureLoss(cond_field='Marker', loss_metric='MCR')(data=predictions)
