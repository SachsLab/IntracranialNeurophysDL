from pathlib import Path
import csv
import h5py
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import pandas as pd


# finger_pos was recorded as (z, -x, -y) in cm. Convert it to x,y,z in mm to match cursor_pos
FING2CURS = np.array([[0, 0, 10], [-10, 0, 0], [0, -10, 0]])
FING2CURS6D = np.eye(6); FING2CURS6D[:3, :3] = FING2CURS


def load_mat(filebase, fast_spike_times=True):
    output = {
        'behav srate': 250.,  # Within machine precision of 1 / np.mean(np.diff(t_vec))
        'spikes srate': 24414.0625
    }

    # Behaviour and spiking data
    with h5py.File(filebase.with_suffix('.mat'), 'r') as f:

        # Behavioural data
        behav = {'Timestamp': f['t'].value.flatten()}
        behav.update({curs: f['cursor_pos'].value[ix, :] for ix, curs in enumerate(['CursorX', 'CursorY'])})

        fing_col_names = ['FingerX', 'FingerY', 'FingerZ']
        if f['finger_pos'].value.shape[0] == 6:
            finger_pose = f['finger_pos'].value.T.dot(FING2CURS6D)
            fing_col_names += ['Azimuth', 'Elevation', 'Roll']
        else:
            finger_pose = f['finger_pos'].value.T.dot(FING2CURS)
        behav.update({col: finger_pose[:, ix] for ix, col in enumerate(fing_col_names)})

        behav.update({targ: f['target_pos'].value[ix, :] for ix, targ in enumerate(['TargetX', 'TargetY'])})

        output['behaviour'] = pd.DataFrame(data=behav)

        # Spiking data.
        chan_names = [''.join(chr(i) for i in f[f['chan_names'].value[0][chan_ix]][:])
                      for chan_ix in range(f['chan_names'].size)]
        n_units, n_chans = f['spikes'].shape
        all_spikes = [f[f['spikes'].value[unit_ix, chan_ix]].value.flatten() for chan_ix in range(n_chans)
                      for unit_ix in range(n_units)]
        temp = np.concatenate(all_spikes)
        spk_range = [np.min(temp[temp > 1]), np.max(temp)]
        t_vec = np.arange(spk_range[0], spk_range[1], 1/output['spikes srate'])

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

        output['spikes'] = {
            'names': np.repeat(chan_names, n_units),
            'units': np.asarray(list(range(n_units)) * n_chans, dtype=int),
            'timestamps': t_vec,
            'data': sparse_mat
        }

    return output


def print_attrs(name, obj):
    print(name)
    if 'keys' in obj.attrs:
        print(list(obj.keys()))
    elif isinstance(obj, h5py.Group):
        for key, val in obj.items():
            print("    %s: %s" % (key, val))
    elif isinstance(obj, h5py.Dataset):
        print(obj.shape)


def get_broadband(filebase):
    with h5py.File(filebase.with_suffix('.nwb'), 'r') as f:
        # f.visititems(print_attrs)
        t_vec = f['acquisition/timeseries/broadband/timestamps'].value.flatten()
        chan_names = f['acquisition/timeseries/broadband/electrode_names'].value
        # data_shape = [len(t_vec), len(chan_names)]
        # data = f['/acquisition/timeseries/broadband/data'].value
        # scale = np.max(np.abs(data))
        # data = (data / scale).astype(np.float16)
        output = {
            't': t_vec,
            'effective srate': 1 / np.median(np.diff(t_vec)),
            'names': chan_names,
            'locs': f['general/extracellular_ephys/electrode_map'].value,
            'data': f['/acquisition/timeseries/broadband/data'].value
        }
    return output


if __name__ == "__main__":
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
        mat_dict = load_mat(_fname)
        _outname = local_dir / row['filename']
        with open(_outname.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(mat_dict, f)
        # Load with...
        # with open(_fname.with_suffix('.pkl'), 'rb') as f:
        #     mat_dict, spikes = pickle.load(f)

        if False and _fname.with_suffix('.nwb').exists():
            bb_dict = get_broadband(_fname)
