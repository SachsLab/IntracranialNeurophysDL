import os
import sys
from pathlib import Path
from indl.lfads.data_helper import *

import jax.numpy as np
from jax import jit, random, vmap, grad, lax
from jax.experimental import optimizers, stax

import indl.lfads.distributions as dists
import indl.lfads.utils as utils
import numpy as onp  # original CPU-backed NumPy
import time
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)


def gru(params, h, x, bfg=0.5):
    """
    Implement the GRU equations.

    Arguments:
    params: dictionary of GRU parameters
    h: np array of hidden state
    x: np array of input
    bfg: bias on forget gate (useful for learning if > 0.0)

    Returns:
    np array of hidden state after GRU update
    """
    hx = np.concatenate([h, x], axis=0)
    ru = sigmoid(np.dot(params['wRUHX'], hx) + params['bRU'])
    r, u = np.split(ru, 2, axis=0)
    rhx = np.concatenate([r * h, x])
    c = np.tanh(np.dot(params['wCHX'], rhx) + params['bC'] + bfg)
    return u * h + (1.0 - u) * c


def run_gru(params, x_t, h0=None):
    """
    Run a GRU module forward in time.

    Arguments:
    params: dictionary of parameters for gru (keys: 'wRUHX', 'bRU', 'wCHX', 'bC') and optionally 'h0'
    x_t: np array data for RNN input with leading dim being time
    h0: initial condition for running rnn, which overwrites param h0

    Returns:
    np array of rnn applied to time data with leading dim being time
    """
    h = h0 if h0 is not None else params['h0']
    h_t = []
    for x in x_t:
        h = gru(params, h, x)
        h_t.append(h)
    return np.array(h_t)


def OverlyVerboseExampleLayerThatDoesNothing(**kwargs):
    """
    DO NOT USE! This is only a guide for making layers.
    Each layer constructor function returns an (init_fun, apply_fun) pair, where
        init_fun: takes an rng key and an input shape and returns an
                  (output_shape, params) pair,
        apply_fun: takes params, inputs, and an rng key and applies the layer.
                   Though I haven't seen any apply_fun with the rng key argument.

    Layers implementing this interface can be composed by jax.experimental.stax.serial
    https://github.com/google/jax/blob/master/jax/experimental/stax.py#L296-L305
    """
    info_needed_to_calc_output = kwargs['something']
    info_needed_to_gen_input_params = kwargs['something else']
    some_fn_to_init_params = lambda rng, y: y  # Just so this code does not break your file.

    def init_fun(rng, input_shape):
        rng1, rng2 = random.split(rng)
        output_shape = input_shape + info_needed_to_calc_output['key1']
        params_1 = some_fn_to_init_params(rng1, info_needed_to_gen_input_params[0])
        params_2 = some_fn_to_init_params(rng2, info_needed_to_gen_input_params[1])
        init_params = params_1, params_2
        return output_shape, init_params

    def apply_fun(params, inputs, **kwargs):
        weights, biases, activation_fn = params
        result = activation_fn(np.dot(weights, inputs) + biases)
        return result

    return init_fun, apply_fun


def gru_params(rng, n, u, ifactor=1.0, hfactor=1.0, hscale=0.0):
    """
    The GRU initialization is common to both the GRU layer and
    twice in the BidirectionalGRU layer, so we have this helper function do it.
    :param rng:
    :param n: hidden state size
    :param u: input size
    :param ifactor: scaling factor for input weights
    :param hfactor: scaling factor for hidden -> hidden weights
    :param hscale: scale on h0 initial condition
    :return:
    """
    rng, keys = utils.keygen(rng, 5)

    ifac = ifactor / np.sqrt(u)
    hfac = hfactor / np.sqrt(n)

    wRUH = random.normal(next(keys), (n + n, n)) * hfac
    wRUX = random.normal(next(keys), (n + n, u)) * ifac
    wRUHX = np.concatenate([wRUH, wRUX], axis=1)

    wCH = random.normal(next(keys), (n, n)) * hfac
    wCX = random.normal(next(keys), (n, u)) * ifac
    wCHX = np.concatenate([wCH, wCX], axis=1)

    return {'h0': random.normal(next(keys), (n,)) * hscale,
            'wRUHX': wRUHX,
            'wCHX': wCHX,
            'bRU': np.zeros((n + n,)),
            'bC': np.zeros((n,))}


def GRU(n_hidden, ifactor=1.0, hfactor=1.0, hscale=0.0):
    """
    GRU Layer
    :param n_hidden: hidden state size
    :param ifactor: scaling factor for input weights
    :param hfactor: scaling factor for hidden -> hidden weights
    :param hscale: scale on h0 initial condition
    :return: (init_fun, apply_fun) tuple
    """

    def init_fun(rng, input_shape):
        u = input_shape[-1]
        output_shape = input_shape[:-1] + (n_hidden,)
        rng, keys = utils.keygen(rng, 1)
        gen_params = gru_params(next(keys), n_hidden, u, ifactor, hfactor, hscale)
        return output_shape, gen_params

    def apply_fun(params, x_t, **kwargs):
        return run_gru(params, x_t)

    return init_fun, apply_fun


def BidirectionalGRU(n_hidden, ifactor=1.0, hfactor=1.0, hscale=0.0):
    """
    Bidirectional GRU Layer.
    :param n_hidden: hidden state size
    :param ifactor: scaling factor for input weights
    :param hfactor: scaling factor for hidden -> hidden weights
    :param hscale: scale on h0 initial condition
    :return: (init_fun, apply_fun) tuple
    """

    def init_fun(rng, input_shape):
        u = input_shape[-1]
        output_shape = (input_shape[0], 2, n_hidden)
        rng, keys = utils.keygen(rng, 2)
        ic_enc_params = {'fwd_rnn': gru_params(next(keys), n_hidden, u, ifactor, hfactor, hscale),
                         'bwd_rnn': gru_params(next(keys), n_hidden, u, ifactor, hfactor, hscale)}
        return output_shape, ic_enc_params

    def apply_fun(params, x_t, **kwargs):
        fwd_enc_t = run_gru(params['fwd_rnn'], x_t)
        bwd_enc_t = np.flipud(run_gru(params['bwd_rnn'], np.flipud(x_t)))
        # full_enc = np.concatenate([fwd_enc_t, bwd_enc_t], axis=1)  # Only used to calc. inferred input.
        enc_ends = np.concatenate([bwd_enc_t[0], fwd_enc_t[-1]], axis=1)
        return enc_ends

    return init_fun, apply_fun


def SampleDistrib(var_min):

    def init_fun(rng, input_shape):
        output_shape = input_shape
        output_shape[0] = output_shape[0] / 2
        # TODO: dists.diagonal_gaussian_params() ?
        return output_shape, {'rng': rng}

    def apply_fun(params, inputs):
        rng, keys = utils.keygen(params['rng'], 1)
        _mean, _logvar = np.split(inputs, 2, axis=0)
        samples = dists.diag_gaussian_sample(next(keys), _mean, _logvar, var_min)
        return samples

    return init_fun, apply_fun


def Generator(n_timesteps, n_factors, dropout_rate):
    def init_fun(rng, input_shape):
        output_shape = (n_factors, n_timesteps)
        rng, keys = utils.keygen(rng, 2)
        # TODO: init params for gru
        gen_gru_params = gru_params(next(keys), gen_dim, ii_dim)
        gen_gru_params = gru_params(next(keys), n_hidden, u)

        # params for dropout: rng key

        init_params = {
            'rng': next(keys),
            'mode': 'train'
        }
        return output_shape, init_params

    def apply_fun(params, inputs, **kwargs):
        rng, keys = utils.keygen(params['rng'], n_timesteps)
        latent_factors = []
        for step_ix in range(n_timesteps):
            g = gru(params, g, 0)
            if params['mode'] == 'train':
                keep = random.bernoulli(rng, dropout_rate, inputs.shape)
                g = np.where(keep, inputs / dropout_rate, 0)
            f = normed_linear(g)
            latent_factors.append(f)
        return latent_factors

    return init_fun, apply_fun


def ReconstructSpikes(params):
    def init_fun(rng, input_shape):
        return output_shape, init_params

    def apply_fun(params, inputs, **kwargs):
        reconstructed_rates = np.exp(inputs)
        # The rates parameterize a Poisson distribution from which spikes are drawn.
        reconstructed_binned_spike_counts = Poisson(reconstructed_rates)
        return reconstructed_binned_spike_counts

    return init_fun, apply_fun


def LFADSEncoderModel(dropout_rate, n_hidden, gen_dim):
    return stax.serial(
        stax.Dropout(dropout_rate),
        BidirectionalGRU(n_hidden),
        stax.Dropout(dropout_rate),
        stax.Dense(2 * gen_dim)
    )


def LFADSDecoderModel(var_min, n_timesteps, n_factors, dropout_rate):
    return stax.serial(
        SampleDistrib(var_min),
        Generator(n_timesteps, n_factors, dropout_rate),
        # TODO: GRU with dropout per-step, and input is state
        # TODO: NormedLinear to transform gru hidden state to factors
        ReconstructSpikes()
    )


if __name__ == '__main__':
    MAX_SEED_INT = 10000000
    SESS_IDX = 7  # Index of recording session we will use. 0:8
    MAX_TRIAL_DUR = 1.7  # This gets rid of about 7% of the slowest trials (long tail distribution)
    BIN_DURATION = 0.020  # Width of window used to bin spikes, in seconds
    P_TRAIN = 0.8  # Proportion of data used for training.

    BATCH_SIZE = 4  # Number of trials in each training step during optimization
    EPOCHS = 10  # Number of loops through the entire data set.

    # LFADS architecture
    N_RNN_UNITS = 60  # Size of RNN output (state)
    ENC_DIM = 128  # encoder dim
    CON_DIM = 128  # controller dim
    GEN_DIM = 128  # generator dim, should be large enough to generate dynamics
    FACTORS_DIM = 32  # factors dim, should be large enough to capture most variance of dynamics

    # Numerical stability
    VAR_MIN = 0.001  # Minimal variance any gaussian can become.

    # Initial state prior parameters
    # the mean is set to zero in the code
    IC_PRIOR_VAR = 0.1  # this is sigma^2 of uninformative prior

    # Optimization Hyperparameters

    # Learning rate HPs
    STEP_SIZE = 0.05  # initial learning rate
    DECAY_FACTOR = 0.995  # learning rate decay param
    DECAY_STEPS = 1  # learning rate decay param

    # Regularization HPs
    P_DROPOUT = 0.05  # Proportion of units to set to 0 on each step.
    keep_rate = 1 - P_DROPOUT
    L2_REG = 2.0e-5  # Parameter regularization strength.

    # Numerical stability HPs
    MAX_GRAD_NORM = 10.0  # gradient clipping above this value

    kl_warmup_start = 50.0  # batch number to start kl warmup, explicitly float
    kl_warmup_end = 250.0  # batch number to be finished with kl warmup, explicitly float
    kl_min = 0.01  # The minimum KL value, non-zero to make sure KL doesn't grow crazy before kicking in.
    kl_max = 1.0

    # Get the data
    datadir = Path.cwd() / 'data' / 'joeyo'
    rng = random.PRNGKey(onp.random.randint(0, MAX_SEED_INT))
    X, Y, X_ax_info, Y_ax_info, targ_times = load_dat_with_vel_accel(datadir, SESS_IDX, trial_dur=MAX_TRIAL_DUR)
    trial_X, trial_tvec, true_bin_dur = bin_and_segment_spike_times(X, X_ax_info, targ_times,
                                                                    nearest_bin_dur=BIN_DURATION,
                                                                    trial_dur=MAX_TRIAL_DUR)
    n_trials, n_timesteps, n_neurons = trial_X.shape
    X_train, X_valid = train_test_split(trial_X, train_size=P_TRAIN)

    # Get the model
    encoder_init, encode = LFADSEncoderModel(P_DROPOUT)
    decoder_init, decode = LFADSDecoderModel()

    # Init the model
    rng, keys = utils.keygen(rng, 2)
    latent_shape, init_encoder_params = encoder_init(next(keys), (BATCH_SIZE, n_timesteps, n_neurons))
    decoded_shape, init_decoder_params = decoder_init(next(keys), (BATCH_SIZE, 0))

    num_batches = int(n_trials * EPOCHS / BATCH_SIZE)  # how many batches do we train

    for binned_spike_counts in batched_data:

        ##################
        ## ENCODER PART ##

        # Randomly drop out spikes. Re-randomized for each time-step!
        do_binned_spike_counts = Dropout(binned_spike_counts)

        # Run the data through a nonlinear recurrent **encoder**. They use a Bidirectional GRU layer.
        # n_neurons -> 2 * ENC_DIM
        full_enc_fwd_bkwd, enc_ends_fwd_last_bkwd_first = BidirectionalGRU(do_binned_spike_counts)
        # AFAICT, full_enc_fwd_bkwd is only used when calculating inferred inputs.

        # Convert the end-points of the RNN output to 'initial conditions', which is the per-trial mean and variance
        # to produce random vectors to encode the trial. This is the 'bottleneck' or 'latent variables' in a VAE.
        # 2 * ENC_DIM -> 2 * GEN_DIM; GEN_DIM for means and GEN_DIM for logvars
        ic_mean_logvar = Dense(Dropout(enc_ends_fwd_last_bkwd_first))

        ##################
        ## DECODER PART ##

        # The initial state used in the next step is a vector randomly drawn from the distribution calculated above.
        g = initial_state = diag_gaussian_sample(ic_mean_logvar)

        # The generator is a GRU that takes as input the initial state.
        # In the tensorflow implementation, this was a slightly specialized GRU: https://github.com/tensorflow/models/blob/c80de2ca3ec34c4d510961b6a604309772fc02ad/research/lfads/lfads.py#L158
        # The output are the "neural modes", aka "latent factors".
        # 2 * GEN_DIM -> FACTORS_DIM
        latent_factors = []
        for step_ix in range(n_bins_per_trial):
            g = GRU(g)
            g = Dropout(g)
            f = normed_linear(g)

            latent_factors.append(f)

        # The factors are transformed to give 'rates' for as many channels as there were neurons in the binned_spike_counts.
        reconstructed_log_rates = Dense(latent_factors)
        reconstructed_rates = exp(reconstructed_log_rates)

        # The rates parameterize a Poisson distribution from which spikes are drawn.
        reconstructed_binned_spike_counts = Poisson(reconstructed_rates)

        ##########
        ## LOSS ##

        log_p_xgz = mean(log_likelihood(binned_spike_counts, reconstructed_binned_spike_counts))
        loss = -log_p_xgz + kl_loss_g0 + l2_loss

        ############
        ## UPDATE ##

        # All of the above would be wrapped in a compiled loss_fn called once per batch,
        # then we can use JAX grad to calculate the gradients of that function.
        gradients = grad(loss_fn)

        # Then the optimizer updates the model weights using the gradients.
        optimizer_update(gradients)