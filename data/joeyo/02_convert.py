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


def get_behav_and_spikes(filename):
    """
    Get behaviour and spiketrain data from mat file.
    :param filename: Full path to mat file.
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

                unit_spike_t_inds = np.searchsorted(t_vec, spike_times)  # timestamp that the spike was BEFORE
                b_shift_down = (spike_times - t_vec[unit_spike_t_inds]) < (-0.5/SPIKES_RATE)  # spikes nearer prev. ts
                unit_spike_t_inds[b_shift_down] -= 1
                spike_t_inds.extend(unit_spike_t_inds)

                spike_unit_inds.extend([chan_ix * n_units + unit_ix] * len(spike_times))
        sparse_dat = np.ones_like(spike_t_inds, dtype=np.bool)
        sparse_mat = csr_matrix((sparse_dat, (spike_unit_inds, spike_t_inds)),
                                shape=(n_chans*n_units, len(t_vec)),
                                dtype=np.bool)
        st_space_ax = npe.SpaceAxis(names=np.repeat(chan_names, n_units),
                                    units=np.tile(np.arange(0, n_units, dtype=np.int8), n_chans))
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
    ROW_RANGE = [13, 19]  # Use [0, np.inf] to process all rows.
    # ROW_RANGE = [34, 35]  # From Ahmadi et al.
    if Path.cwd().stem == 'joeyo':
        import os
        os.chdir('../..')
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
    local_dir = working_dir / 'converted'
    if not local_dir.is_dir():
        local_dir.mkdir()
    print("Saving converted data into {}".format(local_dir))

    for row_ix, row in enumerate(datasets):
        if row_ix < ROW_RANGE[0] or row_ix > ROW_RANGE[1]:
            continue
        print("Converting {}...".format(row['filename']))
        _fname = working_dir / 'download' / row['filename']
        behav_chnk, spikes_chnk = get_behav_and_spikes(_fname.with_suffix('.mat'))
        behav_pkt = npe.Packet({'behav': behav_chnk})
        spk_pkt = npe.Packet({'spiketimes': spikes_chnk})
        # With spiking data, get rid of spikes within 1.0-msec refractory period,
        # get rid of spikes that occur on 30% of all channels on the exact same sample,
        # and bin spikes to resample at the rate nearest 1000.0 that is an integer factor of the input rate.
        spk_pkt = npn.SanitizeSpikeTrain(min_refractory_period=1.0, downsample_rate=1000.,
                                         chan_pcnt_noise_thresh=30., offline_min_spike_rate=0.5)(data=spk_pkt)

        # Convert spike trains to continuous spike rates using a gaussian kernel.
        rates_pkt = npn.InstantaneousEventRate(kernel='gaussian', kernel_parameter=0.05, unit='seconds')(data=spk_pkt)
        rates_pkt = npn.RenameStreams({'spiketimes': 'su_rates'})(data=rates_pkt)

        # And multi-unit activity
        mu_spk_pkt = npn.SimpleSpikeSorting(cluster_method='none')(data=spk_pkt)  # Collapses across units per channel
        mua_rates_pkt = npn.InstantaneousEventRate(kernel='gaussian', kernel_parameter=0.05, unit='seconds')(data=mu_spk_pkt)
        mu_spk_pkt = npn.RenameStreams({'spiketimes': 'mu_spiketimes'})(data=mu_spk_pkt)
        mua_rates_pkt = npn.RenameStreams({'spiketimes': 'mu_rates'})(data=mua_rates_pkt)

        rates_pkt = npn.MergeStreams()(data1=rates_pkt, data2=mua_rates_pkt)
        mua_pkt = None  # Clear from memory.
        spk_pkt = npn.MergeStreams()(data1=spk_pkt, data2=mu_spk_pkt)
        mu_spk_pkt = None

        # Resample spike rates at same rate as behaviour. Includes anti-aliasing filter.
        # This is unnecessary because we are interpolating below. Anti-aliasing not required because of smoothing above.
        # rates_pkt = npn.Resample(rate=behav_pkt.chunks['behav'].block.axes['time'].nominal_rate)(data=rates_pkt)

        # Slice rates and behaviour to minimum over-lapping time domain
        behav_times = behav_pkt.chunks['behav'].block.axes['time'].times
        sua_times = rates_pkt.chunks['su_rates'].block.axes['time'].times
        spk_times = spk_pkt.chunks['spiketimes'].block.axes['time'].times
        new_time_range = (max(behav_times[0], sua_times[0], spk_times[0]),
                          min(behav_times[-1], sua_times[-1], spk_times[-1]))
        new_t_range_str = str(new_time_range[0]) + '...' + str(new_time_range[-1])

        rates_pkt = npn.SelectRange(axis='time', selection=new_t_range_str, unit='seconds')(data=rates_pkt)
        behav_pkt = npn.SelectRange(axis='time', selection=new_t_range_str, unit='seconds')(data=behav_pkt)
        spk_pkt = npn.SelectRange(axis='time', selection=new_t_range_str, unit='seconds')(data=spk_pkt)

        # Linearly interpolate rates at behaviour timestamps.
        # Note that here extrapolation is < 1/fs outside the data range because we used SelectRange above,
        # and the rates have all been smoothed with a Gaussian kernel, so it should be quite safe.
        rates_pkt = npn.Interpolate(new_points=behav_pkt.chunks['behav'].block.axes['time'].times,
                                    kind='linear', allow_extrapolation=True)(data=rates_pkt)
        rates_pkt.chunks['su_rates'].block.axes['time'].nominal_rate =\
            behav_pkt.chunks['behav'].block.axes['time'].nominal_rate
        rates_pkt.chunks['mu_rates'].block.axes['time'].nominal_rate =\
            behav_pkt.chunks['behav'].block.axes['time'].nominal_rate

        # Merge the streams together.
        data_pkt = npn.MergeStreams()(data1=behav_pkt, data2=spk_pkt, data3=rates_pkt)
        behav_pkt = spk_pkt = rates_pkt = None  # Clear memory

        # If this dataset has broadband data then let's save a 1 kHz version of that (i.e. LFPs)
        # Skip this for now. I think maybe it was recorded with 500-5000 bandpass, which makes it useless for LFPs.
        # TODO: Inspect frequency transform.
        if False and _fname.with_suffix('.nwb').exists():
            print("Getting LFP data from {}...".format(_fname.with_suffix('.nwb')))
            bb_dict = get_broadband(_fname.with_suffix('.nwb'))
            time_ax = npe.TimeAxis(times=bb_dict['t'], nominal_rate=bb_dict['effective srate'])
            space_ax = npe.SpaceAxis(names=bb_dict['names'], positions=bb_dict['locs'])
            lfp_pkt = npe.Packet({'lfps': npe.Chunk(block=npe.Block(data=bb_dict['data'], axes=(time_ax, space_ax)),
                                                    props={npe.Flags.has_markers: False, npe.Flags.is_signal: True})})
            lfp_pkt = npn.Resample(rate=1000.0)(data=lfp_pkt)  # Includes anti-alias filtering
            # Common-average referencing.
            lfp_pkt = npn.Rereferencing(axis='space', reference_range=':')(data=lfp_pkt)
            # High-pass filtering
            lfp_pkt = npn.IIRFilter(frequencies=[0.05, 0.5], mode='highpass', offline_filtfilt=True)(data=lfp_pkt)
            # Notch filter out powerline noise. TODO: Also filter out harmonics with a Comb filter.
            lfp_pkt = npn.IIRFilter(frequencies=[57, 63], mode='bandstop', offline_filtfilt=True)(data=lfp_pkt)
            # Slice LFP to same timespan as behaviour and rates
            behav_ts = data_pkt.chunks['behav'].block.axes['time'].times
            new_lfp_range = str(behav_ts[0] - (5 / 1000)) + ':' + str(behav_ts[-1] + 1 / 1000)
            lfp_pkt = npn.SelectRange(axis='time', selection=new_lfp_range, unit='seconds')(data=lfp_pkt)

            data_pkt = npn.MergeStreams()(data1=data_pkt, data2=lfp_pkt)

        npn.ExportH5(filename=str((local_dir / row['filename']).with_suffix('.h5')))(data=data_pkt)

# from the data/joeyo folder
# Once only: kaggle datasets init -p converted
# Once only: kaggle datasets create -p converted
# On new computer: kaggle datasets metadata -p converted cboulay/joeyo_nhp_reach_mea
# On updates: kaggle datasets version -m "Updated data." -p converted
