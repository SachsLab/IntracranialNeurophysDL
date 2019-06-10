from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.utils.fileio import load_joeyo_reaching
from data.utils.KalmanFilter import ManualKalmanFilter as mkf
from filterpy.kalman.UKF import UnscentedKalmanFilter as ukf
from filterpy.kalman import MerweScaledSigmaPoints


# Hyperparameters
SESS_IDX = 0          # Index of recording session we will use. 0:8
BIN_DURATION = 0.250  # Width of window used to bin spikes, in seconds
N_TAPS = 1            # Number of bins of history used in a sequence.
P_TRAIN = 0.8         # Proportion of data used for training.
BATCH_SIZE = 32       # Number of sequences in each training step.
P_DROPOUT = 0.05      # Proportion of units to set to 0 on each step.
N_RNN_UNITS = 60      # Size of RNN output (state)
L2_REG = 1.7e-5       # Parameter regularization strength.
STATEFUL = False      # Whether or not to keep state between sequences (True is not tested)
EPOCHS = 10           # Number of loops through the entire data set.

# Prepare Data


def load_dat_with_vel_accel(datadir, sess_idx):
    BEHAV_CHANS = ['CursorX', 'CursorY']
    sess_names = ['indy_201' + _ for _ in ['60921_01', '60927_04', '60927_06', '60930_02', '60930_05', '61005_06',
                                           '61006_02', '60124_01', '60127_03']]
    X, Y, X_ax_info, Y_ax_info = load_joeyo_reaching(datadir, sess_names[sess_idx], x_chunk='mu_spiketimes')

    # Slice Y to only keep required behaviour data (cursor position)
    b_keep_y_chans = np.in1d(Y_ax_info['channel_names'], BEHAV_CHANS)
    Y = Y[b_keep_y_chans, :]
    Y_ax_info['channel_names'] = [_ for _ in Y_ax_info['channel_names'] if _ in BEHAV_CHANS]

    # Calculate discrete derivative and double-derivative to get velocity and acceleration.
    vel = np.diff(Y, axis=1)
    vel = np.concatenate((vel[:, 0][:, None], vel),
                         axis=1)  # Assume velocity was constant across the first two samples.
    accel = np.concatenate(([[0], [0]], np.diff(vel, axis=1)), axis=1)  # Assume accel was 0 in the first sample.
    Y = np.concatenate((Y, vel, accel), axis=0)
    Y_ax_info['channel_names'] += ['VelX', 'VelY', 'AccX', 'AccY']

    return X, Y, X_ax_info, Y_ax_info

datadir = Path.cwd()
X, Y, X_ax_info, Y_ax_info = load_dat_with_vel_accel(datadir, SESS_IDX)

# Bin spike times
def bin_spike_times(X, X_ax_info, bin_duration=0.256, bin_step_dur=0.004):
    bin_samples = int(np.ceil(bin_duration * X_ax_info['fs']))
    bin_starts_t = np.arange(X_ax_info['timestamps'][0], X_ax_info['timestamps'][-1], bin_step_dur)
    bin_starts_idx = np.searchsorted(X_ax_info['timestamps'], bin_starts_t)

    # Only keep bins that do not extend beyond the data limit.
    b_full_bins = bin_starts_idx <= (X.shape[-1] - bin_samples)
    bin_starts_idx = bin_starts_idx[b_full_bins]
    bin_starts_t = bin_starts_t[b_full_bins]

    # The next chunk of code counts the number of spikes in each bin.
    # Create array of indices to reslice the raster data
    bin_ix = np.arange(bin_samples)[:, None] + bin_starts_idx[None, :]
    # Create buffer to hold the dense raster data
    _temp = np.zeros(X[0].shape, dtype=bool)
    # Create output variable to hold spike counts per bin
    _X = np.zeros((len(bin_starts_idx), X.shape[0]), dtype=np.int32)
    for chan_ix in range(X.shape[0]):
        _X[:, chan_ix] = np.sum(X[chan_ix].toarray(out=_temp)[0][bin_ix], axis=0)
    _X = _X / bin_duration

    return _X.astype(np.float32), bin_starts_t, bin_samples


_X, bin_starts_t, bin_samples = bin_spike_times(X, X_ax_info, bin_duration=BIN_DURATION,
                                                bin_step_dur=(1 / Y_ax_info['fs']))

# Aligning the data
def get_binned_rates_with_history(_X, Y, X_ax_info, bin_starts_t, bin_samples, n_taps=3):
    bin_stops_t = bin_starts_t + bin_samples / X_ax_info['fs']
    bin_stops_t = bin_stops_t[(N_TAPS - 1):]

    _X_tapped = np.lib.stride_tricks.as_strided(_X, shape=(len(bin_stops_t), N_TAPS, _X.shape[-1]),
                                                strides=(_X.strides[-2], _X.strides[-2], _X.strides[-1]))

    b_keep_y = Y_ax_info['timestamps'] > bin_stops_t[0]
    n_extra_y = np.sum(b_keep_y) - len(bin_stops_t)
    if n_extra_y > 0:
        b_keep_y[-n_extra_y] = False
    _Y = Y[:, b_keep_y].T

    _X_tapped = _X_tapped[:_Y.shape[0], :, :]
    bin_stops_t = bin_stops_t[:_Y.shape[0]]

    return _X_tapped, _Y.astype(np.float32), bin_stops_t


_X_tapped, _Y, bin_stops_t = get_binned_rates_with_history(_X, Y, X_ax_info, bin_starts_t, bin_samples, n_taps=N_TAPS)

# Training and test data for Kalman filter
z = np.squeeze(_X_tapped).T
x = _Y[:, :4].T
training_size = int(0.8 * np.size(x, 1))
x_training = x[:, :training_size]
z_training = z[:, :training_size]
x_test = x[:, training_size:]
z_test = z[:, training_size:]

# The empty array to store in our predicted states later
x_predict = np.zeros_like(x_test)
# Instantiating a Kalman filter object (Inputs are state and neural training data
mykf = mkf(x=x_training, z=z_training)
# just because it's slow to do the whole test data
rng = int(np.size(x_test, 1))
# Estimating a state with each of neural test data
for i in range(rng):
    mykf.predict()
    z_in = z_test[:, i]
    x_predict[:, i] = np.reshape(mykf.update(z_in), (mykf.m, ))
    if i%50 == 0:
        print('KF - Step: ' + str(i) + ' out of ' + str(rng))

# Unscented Kalman Filter
states = np.concatenate((_Y.T, z))
training_size = int(0.8 * np.size(states, 1))
x_training = states[:6, :training_size]
z_training = states[6:, :training_size]
x_test = states[:6, training_size:]
z_test = states[6:, training_size:]
dim_x = np.size(_Y, 1)
dim_z = np.size(z, 0)

# Initializing matrices
x1 = x_training[:, :-1]
x2 = x_training[:, 1:]
temp1 = np.dot(x2, x1.T)
temp2 = np.linalg.inv(np.dot(x1, x1.T))
F = np.dot(temp1, temp2)
# Q = ((X2 - F.X1).(X2 - FX1)^T) / (M-1)
temp = x2 - np.dot(F, x1)
Q = np.dot(temp, temp.T) / (dim_x - 1)
# H = Z.X^T.(X.X^T)^-1
temp1 = np.dot(z_training, x_training.T)
temp2 = np.linalg.inv(np.dot(x_training, x_training.T))
H = np.dot(temp1, temp2)
# R = ((Z - H.X).(Z - H.X)^T) / M
Z_HX = z_training - np.dot(H, x_training)
temp = np.dot(Z_HX, Z_HX.T)
R = np.divide(temp, dim_x)



def transition(state, time_step):
    output = np.zeros_like(state)
    cursor = state[:6]
    output[:6] = np.dot(F, cursor)
    output[6:] = state[6:]
    return output


def observation (state):
    return H @ state[:6]


points = MerweScaledSigmaPoints(np.size(states, 0), alpha=1., beta=2., kappa=0)
myukf = ukf(dim_x= dim_x + dim_z, dim_z=dim_z, dt=0.001, fx= transition, hx=observation, points=points)
initial_state = states[:, 0]
myukf.x = initial_state
myukf.R = R
myukf.Q = np.eye(dim_x + dim_z)

# Estimation loop
rng = np.size(states, 1) - training_size
ukf_predict = np.zeros((dim_x + dim_z, rng))
for i in range(training_size, np.size(states, 1)):
    firing_rates = states[6:, i]
    myukf.predict(dt=0.001)
    myukf.update(firing_rates)
    ukf_predict[:, i - training_size] = myukf.x
    if (i - training_size) % 50 == 0:
        print('UKF - Step: ' + str(i - training_size) + ' out of ' + str(rng))

# Plotting the results
plt.subplot(4,1,1)
plt.plot(ukf_predict[0, :], label="UKF")
plt.plot(x_predict[0, :], label="KF")
plt.plot(states[0, -rng:], label="actual")
plt.title("X-Position")
plt.legend()
plt.subplot(4,1,2)
plt.plot(ukf_predict[1, :], label="UKF")
plt.plot(x_predict[1, :], label="KF")
plt.plot(states[1, -rng:], label="actual")
plt.title("Y-Position")
plt.legend()
plt.subplot(4,1,3)
plt.plot(ukf_predict[2, :], label="UKF")
plt.plot(x_predict[2, :], label="KF")
plt.plot(states[2, -rng:], label="actual")
plt.title("X-Velocity")
plt.legend()
plt.subplot(4,1,4)
plt.plot(ukf_predict[3, :], label="UKF")
plt.plot(x_predict[3, :], label="KF")
plt.plot(states[3, -rng:], label="actual")
plt.title("Y-Velocity")
plt.legend()
plt.show()
