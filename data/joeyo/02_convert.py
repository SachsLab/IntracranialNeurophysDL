from pathlib import Path
import csv
import h5py
import numpy as np
from scipy.sparse import csr_matrix
import neuropype.engine as npe
import neuropype.nodes as npn
import logging


logging.basicConfig(level=logging.DEBUG)
# finger_pos was recorded as (z, -x, -y) in cm. Convert it to x,y,z in mm to match cursor_pos
FING2CURS = np.array([[0, 0, 10], [-10, 0, 0], [0, -10, 0]])
FING2CURS6D = np.eye(6); FING2CURS6D[:3, :3] = FING2CURS
BEHAV_RATE = 250.  # Within machine precision of 1 / np.mean(np.diff(t_vec))
SPIKES_RATE = 24414.0625


def get_behav_and_spikes(filename, fast_spike_times=True):
    """
    Get behaviour and spiketrain data from mat file.
    :param filename: Full path to mat file.
    :param fast_spike_times: True to use np.searchsorted. This is much faster
        but less accurate. Each spike might be off by one sample in either direction.
    :return: (behaviour, spiketrain) - Each is a NeuroPype Chunk.
    """

    # Behaviour and spiking data
    with h5py.File(filename, 'r') as f:
        # Behavioural data
        fing_col_names = ['FingerX', 'FingerY', 'FingerZ']
        if f['finger_pos'][()].shape[0] == 6:
            finger_pose = f['finger_pos'][()].T.dot(FING2CURS6D)
            fing_col_names += ['Azimuth', 'Elevation', 'Roll']
        else:
            finger_pose = f['finger_pos'][()].T.dot(FING2CURS)
        behav = npe.Chunk(block=npe.Block(
            data=np.concatenate((f['cursor_pos'], finger_pose.T, f['target_pos']), axis=0),
            axes=(npe.SpaceAxis(names=['CursorX', 'CursorY'] + fing_col_names + ['TargetX', 'TargetY']),
                  npe.TimeAxis(times=f['t'][()].flatten(), nominal_rate=BEHAV_RATE))),
            props={npe.Flags.has_markers: False, npe.Flags.is_signal: True})

        # Spiking data.
        chan_names = [''.join(chr(i) for i in f[f['chan_names'][0, chan_ix]][:])
                      for chan_ix in range(f['chan_names'].size)]
        n_units, n_chans = f['spikes'].shape
        all_spikes = [f[f['spikes'][unit_ix, chan_ix]][()].flatten() for chan_ix in range(n_chans)
                      for unit_ix in range(n_units)]
        # Inspect all_spikes to get the time range of the data
        temp = np.concatenate(all_spikes)
        spk_range = [np.min(temp[temp > 1]), np.max(temp)]
        t_vec = np.arange(spk_range[0], spk_range[1], 1/SPIKES_RATE)
        # Get the vectors for timestamp indices and channelxunit indices needed to create the sparse matrix.
        spike_t_inds = []
        spike_unit_inds = []
        for chan_ix in range(n_chans):
            for unit_ix in range(n_units):
                st_ix = chan_ix * n_units + unit_ix
                spike_times = all_spikes[st_ix][np.logical_and(all_spikes[st_ix] >= t_vec[0],
                                                               all_spikes[st_ix] <= t_vec[-1])]
                if fast_spike_times:
                    # Faster but less accurate. Each spike might be off by one sample in either direction.
                    spike_t_inds.extend(np.searchsorted(t_vec, spike_times))
                else:
                    # Slower but puts the spike on the nearest sample time.
                    spike_t_inds.extend([np.argmin(np.abs(st - t_vec)) for st in spike_times])
                spike_unit_inds.extend([chan_ix * n_units + unit_ix] * len(spike_times))
        sparse_dat = np.ones_like(spike_t_inds, dtype=np.bool)
        sparse_mat = csr_matrix((sparse_dat, (spike_unit_inds, spike_t_inds)),
                                shape=(n_chans*n_units, len(t_vec)),
                                dtype=np.bool)
        np.asarray(list(range(n_units)) * n_chans, dtype=int)
        st_space_ax = npe.SpaceAxis(names=np.repeat(chan_names, n_units),
                                    units=np.tile(np.arange(0, 3, dtype=np.int8), n_chans))
        st_time_ax = npe.TimeAxis(times=t_vec, nominal_rate=SPIKES_RATE)
        spktrain = npe.Chunk(block=npe.Block(data=sparse_mat, axes=(st_space_ax, st_time_ax)),
                             props={npe.Flags.has_markers: False, npe.Flags.is_sparse: True, npe.Flags.is_signal: True})

    return behav, spktrain


def print_attrs(name, obj):
    print(name)
    if 'keys' in obj.attrs:
        print(list(obj.keys()))
    elif isinstance(obj, h5py.Group):
        for key, val in obj.items():
            print("    %s: %s" % (key, val))
    elif isinstance(obj, h5py.Dataset):
        print(obj.shape)


def get_broadband(filename):
    with h5py.File(filename, 'r') as f:
        # f.visititems(print_attrs)
        t_vec = f['acquisition/timeseries/broadband/timestamps'][()].flatten()
        chan_names = f['acquisition/timeseries/broadband/electrode_names'][()]
        # data_shape = [len(t_vec), len(chan_names)]
        # data = f['/acquisition/timeseries/broadband/data'][()]
        # scale = np.max(np.abs(data))
        # data = (data / scale).astype(np.float16)
        output = {
            't': t_vec,
            'effective srate': 1 / np.median(np.diff(t_vec)),
            'names': chan_names,
            'locs': f['general/extracellular_ephys/electrode_map'][()],
            'data': f['/acquisition/timeseries/broadband/data'][()]
        }
    return output


if __name__ == "__main__":
    # Load a csv file that describes the datasets.
    working_dir = Path.cwd() / 'data' / 'joeyo'
    datasets_file = working_dir / 'datasets.csv'
    # Get the list of datasets to convert
    datasets = []
    with open(datasets_file) as csvfile:
        datasetreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in datasetreader:
            datasets.append(row)

    # Create a local folder to store the data
    local_dir = working_dir / 'convert'
    if not local_dir.is_dir():
        local_dir.mkdir()
    print("Saving converted data into {}".format(local_dir))

    for row in datasets:
        print("Converting {}...".format(row['filename']))
        _fname = working_dir / 'download' / row['filename']
        behav_chnk, spikes_chnk = get_behav_and_spikes(_fname.with_suffix('.mat'))
        behav_pkt = npe.Packet({'behav': behav_chnk})
        spk_pkt = npe.Packet({'spikerates': spikes_chnk})
        # With spiking data, get rid of spikes within 1.0-msec refractory period,
        # get rid of spikes that occur on 30% of all channels on the exact same sample,
        # and bin spikes to resample at the rate nearest 1000.0 that is an integer factor of the input rate.
        spk_pkt = npn.SanitizeSpikeTrain(min_refractory_period=1.0, downsample_rate=1000.,
                                         chan_pcnt_noise_thresh=30., offline_min_spike_rate=0.01)(data=spk_pkt)
        # Convert spike trains to continuous spike rates using a 0.05-second gaussian kernel.
        spk_pkt = npn.InstantaneousEventRate(kernel='gaussian', kernel_parameter=0.05, unit='seconds')(data=spk_pkt)
        # Resample spike rates at the same samples as the behavior.
        spk_pkt = npn.Resample(rate=250.0)(data=spk_pkt)
        spk_pkt = npn.Interpolate(new_points=behav_chnk.block.axes[npe.time].times, kind='nearest')(data=spk_pkt)
        # Merge the two streams together and save as H5.
        data_pkt = npn.MergeStreams()(data1=behav_pkt, data2=spk_pkt)
        npn.ExportH5(filename=str((local_dir / row['filename']).with_suffix('.h5')))(data=data_pkt)

        if False and _fname.with_suffix('.nwb').exists():
            bb_dict = get_broadband(_fname.with_suffix('.nwb'))
