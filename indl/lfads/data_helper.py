import numpy as onp
from data.utils.fileio import load_joeyo_reaching


def load_dat_with_vel_accel(datadir, sess_idx, x_chunk='mu_spiketimes', trial_dur=1.7):
    BEHAV_CHANS = ['CursorX', 'CursorY']
    sess_names = ['indy_201' + _ for _ in ['60921_01', '60927_04', '60927_06', '60930_02', '60930_05', '61005_06',
                                           '61006_02', '70124_01', '70127_03']]
    X, Y, X_ax_info, Y_ax_info = load_joeyo_reaching(datadir, sess_names[sess_idx], x_chunk=x_chunk)

    # Determine target onset times that we will keep.
    targ_ch_ix = onp.where(onp.in1d(Y_ax_info['channel_names'], ['TargetX', 'TargetY']))[0]
    b_targ_onset = onp.hstack((True, onp.any(onp.diff(Y[targ_ch_ix]) != 0, axis=0)))
    targ_onset_times = Y_ax_info['timestamps'][b_targ_onset]
    b_short_trials = onp.hstack((onp.diff(targ_onset_times) <= trial_dur, False))
    print("Keeping {} of {} trials ({} %) shorter than {} s.".format(onp.sum(b_short_trials),
                                                                     b_short_trials.size,
                                                                     100*onp.sum(b_short_trials) / b_short_trials.size,
                                                                     trial_dur))
    targ_onset_times = targ_onset_times[b_short_trials]

    # Identify channels with required behaviour data (cursor position)
    b_keep_y_chans = onp.in1d(Y_ax_info['channel_names'], BEHAV_CHANS)

    # Save target vectors per trial
    first_pos = Y[b_keep_y_chans][:, b_targ_onset][:, b_short_trials]
    targ_pos = Y[targ_ch_ix][:, b_targ_onset][:, b_short_trials]
    targ_vec = targ_pos - first_pos

    # Slice Y to only keep required behaviour data (cursor position)
    Y = Y[b_keep_y_chans, :]
    Y_ax_info['channel_names'] = [_ for _ in Y_ax_info['channel_names'] if _ in BEHAV_CHANS]

    # Calculate discrete derivative and double-derivative to get velocity and acceleration.
    vel = onp.diff(Y, axis=1)
    vel = onp.concatenate((vel[:, 0][:, None], vel), axis=1)  # Assume velocity was constant across the first two samples.
    accel = onp.concatenate(([[0], [0]], onp.diff(vel, axis=1)), axis=1)  # Assume accel was 0 in the first sample.
    Y = onp.concatenate((Y, vel, accel), axis=0)
    Y_ax_info['channel_names'] += ['VelX', 'VelY', 'AccX', 'AccY']

    return X, Y, X_ax_info, Y_ax_info, targ_onset_times, targ_vec


def bin_and_segment_spike_times(X, X_ax_info, targ_onset_times,
                                nearest_bin_dur=0.005, nearest_bin_step_dur=None, trial_dur=1.7):

    # We'll use an integer number of samples per bin: the next highest required to get the requested nearest_bin_dur
    samps_per_bin = int(onp.ceil(nearest_bin_dur * X_ax_info['fs']))
    true_bin_dur = samps_per_bin / X_ax_info['fs']
    print("Actual bin duration: {}".format(true_bin_dur))

    # Similar for number of samples per bin-step.
    if nearest_bin_step_dur is not None:
        samps_per_step = int(onp.ceil(nearest_bin_step_dur * X_ax_info['fs']))
        print("Actual bin duration: {}".format(samps_per_bin / X_ax_info['fs']))
    else:
        # No overlap
        samps_per_step = samps_per_bin

    # Get the indices of each bin-start
    bin_starts_idx = onp.arange(0, X.shape[-1], samps_per_step)
    b_full_bins = bin_starts_idx <= (X.shape[-1] - samps_per_bin)
    bin_starts_idx = bin_starts_idx[b_full_bins]

    # The next chunk of code counts the number of spikes in each bin.
    # -Create array of indices to reslice the raster data
    bin_ix = onp.arange(samps_per_bin)[:, None] + bin_starts_idx[None, :]
    # -Create buffer to hold the dense raster data
    _temp = onp.zeros(X[0].shape, dtype=bool)
    # -Preallocate _X to hold spike counts per bin
    _X = onp.zeros((len(bin_starts_idx), X.shape[0]), dtype=onp.int32)
    for chan_ix in range(X.shape[0]):
        _X[:, chan_ix] = onp.sum(X[chan_ix].toarray(out=_temp)[0][bin_ix], axis=0)

    # Now that our data are binned, let's slice it up into trials.
    bins_per_trial = int(trial_dur / true_bin_dur)
    trial_X = onp.zeros((len(targ_onset_times), bins_per_trial, _X.shape[-1]))

    bin_stops_t = X_ax_info['timestamps'][bin_starts_idx + samps_per_bin - 1]
    trial_starts_idx = onp.searchsorted(bin_stops_t, targ_onset_times)
    for trial_ix, t_start_idx in enumerate(trial_starts_idx):
        trial_X[trial_ix] = _X[t_start_idx:t_start_idx+bins_per_trial, :]

    in_trial_tvec = onp.arange(bins_per_trial) * true_bin_dur

    return trial_X, in_trial_tvec, true_bin_dur
