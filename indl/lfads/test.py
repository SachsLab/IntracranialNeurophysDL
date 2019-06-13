from pathlib import Path
from indl.lfads.data_helper import *

import jax.numpy as np
from jax import jit, random, vmap, grad, lax
from jax.experimental import optimizers, stax

import indl.lfads.distributions as dists
import indl.lfads.utils as utils
import numpy as onp  # original CPU-backed NumPy
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


def run_gru(params, x_t, h0=None, keep_rate=1.0, rng=None):
    """
    Run a GRU module forward in time.

    Arguments:
    params: dictionary of parameters for gru (keys: 'wRUHX', 'bRU', 'wCHX', 'bC') and optionally 'h0'
    x_t: np array data for RNN input with leading dim being time
    h0: initial condition for running rnn, which overwrites param h0
    keep_rate:
    rng:

    Returns:
    np array of rnn applied to time data with leading dim being time
    """
    if keep_rate < 1 and rng is None:
        msg = "Dropout requires rng key."
        raise ValueError(msg)
    rng, keys = utils.keygen(rng, len(x_t))
    h = h0 if h0 is not None else params['h0']
    h_t = []
    for x in x_t:
        h = gru(params, h, x)
        # Do dropout on hidden state
        keep = random.bernoulli(next(keys), keep_rate, h.shape)
        h = np.where(keep, h / keep_rate, 0)
        h_t.append(h)

    return np.array(h_t)


def gru_params(rng, n, u, ifactor=1.0, hfactor=1.0, hscale=0.0):
    """
    Helper function for GRU parameter initialization.
    Used twice in the BidirectionalGRU (encoder) and once in the FreeEvolveGRU (decoder).
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


def BidirectionalGRU(n_hidden):
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
        output_shape = input_shape[:-2] + (2 * n_hidden,)
        rng, keys = utils.keygen(rng, 2)
        ic_enc_params = {'fwd_rnn': gru_params(next(keys), n_hidden, u),
                         'bwd_rnn': gru_params(next(keys), n_hidden, u)}
        return output_shape, ic_enc_params

    def apply_fun(params, x_t):
        fwd_enc_t = run_gru(params['fwd_rnn'], x_t)
        bwd_enc_t = np.flipud(run_gru(params['bwd_rnn'], np.flipud(x_t)))
        enc_ends = np.concatenate([bwd_enc_t[0], fwd_enc_t[-1]], axis=1)
        return enc_ends

    return init_fun, apply_fun


def FreeEvolveGRU(n_hidden, evolve_steps=1, keep_rate=1.0):
    """
    FreeEvolveGRU: The input to this layer is the 'initial conditions' of the GRU,
    and the GRU will evolve for n_timesteps with 0 as the input at each time step.
    :param n_hidden: hidden state size
    :param keep_rate: proportion of gru-outputs to keep (1-dropout)
    :param evolve_steps: number of steps to evolve the output with zero-input.
    :return: (init_fun, apply_fun) tuple
    """
    x0 = np.zeros((evolve_steps, 1))

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (n_hidden,)
        rng, keys = utils.keygen(rng, 1)
        gen_params = gru_params(next(keys), n_hidden, 1)
        # TODO: Modify params so x weights are all 0.
        return output_shape, gen_params

    def apply_fun(params, x_t, rng=None):
        return run_gru(params, x0, h0=x_t, keep_rate=keep_rate, rng=rng)

    return init_fun, apply_fun


def SampleDistrib(var_min):

    def init_fun(rng, input_shape):
        output_shape = input_shape
        output_shape[0] = output_shape[0] / 2
        return output_shape, None

    def apply_fun(params, inputs, rng=None):
        if rng is None:
            msg = "SampleDistrib apply_fun requires rng key."
            raise ValueError(msg)
        rng, keys = utils.keygen(rng, 1)
        _mean, _logvar = np.split(inputs, 2, axis=0)
        samples = dists.diag_gaussian_sample(next(keys), _mean, _logvar, var_min)
        return samples

    return init_fun, apply_fun


def NormedLinear(output_size, ifactor=1.0):

    def init_fun(rng, input_shape):
        u = input_shape[-1]
        key, keys = utils.keygen(rng, 1)
        ifac = ifactor / np.sqrt(u)
        params = {'w': random.normal(next(keys), (output_size, u)) * ifac}
        output_shape = input_shape[:-1] + (output_size,)
        return output_shape, params

    def apply_fun(params, inputs):
        w = params['w']
        w_row_norms = np.sqrt(np.sum(w ** 2, axis=1, keepdims=True))
        w = w / w_row_norms
        return np.dot(w, inputs)

    return init_fun, apply_fun


def LFADSEncoderModel(keep_rate, n_hidden, latent_dim):
    """
    :param keep_rate:
    :param n_hidden:
    :param latent_dim:
    :return: Bottlenecked latent variables (aka initial conditions): means and logvars
    """
    return stax.serial(
        stax.Dropout(keep_rate),
        BidirectionalGRU(n_hidden),
        stax.Dropout(keep_rate),
        stax.Dense(2 * latent_dim)
    )


def LFADSDecoderModel(var_min, keep_rate, n_hidden, n_timesteps, n_factors, n_neurons):
    """
    :param var_min:
    :param keep_rate:
    :param n_hidden:
    :param n_timesteps:
    :param n_factors:
    :param n_neurons:
    :return: Log-rates
    """
    return stax.serial(
        SampleDistrib(var_min),
        FreeEvolveGRU(n_hidden, keep_rate=keep_rate, evolve_steps=n_timesteps),
        NormedLinear(n_factors),  # Returns factor log-rates
        stax.Dense(n_neurons),  # Factors to Neurons
    )


def lfads_losses(params, lfads_hps, key, x_bxt, kl_scale, keep_rate):
    """Compute the training loss of the LFADS autoencoder

    Arguments:
      params: a dictionary of LFADS parameters
      lfads_hps: a dictionary of LFADS hyperparameters
      key: random.PRNGKey for random bits
      x_bxt: np array of input with leading dims being batch and time
      keep_rate: dropout keep rate
      kl_scale: scale on KL

    Returns:
      a dictionary of all losses, including the key 'total' used for optimization
    """
    # ic_prior = {'mean': 0.0 * np.ones((gen_dim,)),
    #          'logvar': np.log(lfads_hps['ic_prior_var']) * np.ones((gen_dim,))}

    B = lfads_hps['batch_size']
    key, skeys = utils.keygen(key, 2)
    keys = random.split(next(skeys), B)
    lfads = batch_lfads(params, lfads_hps, keys, x_bxt, keep_rate)

    # Sum over time and state dims, average over batch.
    # KL - g0
    ic_post_mean_b = lfads['ic_mean']
    ic_post_logvar_b = lfads['ic_logvar']
    kl_loss_g0_b = dists.batch_kl_gauss_gauss(ic_post_mean_b, ic_post_logvar_b,
                                              params['ic_prior'],
                                              lfads_hps['var_min'])
    kl_loss_g0_prescale = np.sum(kl_loss_g0_b) / B
    kl_loss_g0 = kl_scale * kl_loss_g0_prescale

    # KL - Inferred input
    ii_post_mean_bxt = lfads['ii_mean_t']
    ii_post_var_bxt = lfads['ii_logvar_t']
    keys = random.split(next(skeys), B)
    kl_loss_ii_b = dists.batch_kl_gauss_ar1(keys, ii_post_mean_bxt,
                                            ii_post_var_bxt, params['ii_prior'],
                                            lfads_hps['var_min'])
    kl_loss_ii_prescale = np.sum(kl_loss_ii_b) / B
    kl_loss_ii = kl_scale * kl_loss_ii_prescale

    # Log-likelihood of data given latents.
    lograte_bxt = lfads['lograte_t']
    log_p_xgz = np.sum(dists.poisson_log_likelihood(x_bxt, lograte_bxt)) / B

    # L2
    l2reg = lfads_hps['l2reg']
    l2_loss = l2reg * optimizers.l2_norm(params) ** 2

    loss = -log_p_xgz + kl_loss_g0 + kl_loss_ii + l2_loss
    all_losses = {'total': loss, 'nlog_p_xgz': -log_p_xgz,
                  'kl_g0': kl_loss_g0, 'kl_g0_prescale': kl_loss_g0_prescale,
                  'kl_ii': kl_loss_ii, 'kl_ii_prescale': kl_loss_ii_prescale,
                  'l2': l2_loss}
    return all_losses


if __name__ == '__main__':

    # Hyperparameters #

    MAX_SEED_INT = 10000000
    SESS_IDX = 7  # Index of recording session we will use. 0:8
    MAX_TRIAL_DUR = 1.7  # This gets rid of about 7% of the slowest trials (long tail distribution)
    BIN_DURATION = 0.020  # Width of window used to bin spikes, in seconds
    P_TRAIN = 0.8  # Proportion of data used for training.

    BATCH_SIZE = 4  # Number of trials in each training step during optimization
    EPOCHS = 10  # Number of loops through the entire data set.

    # LFADS architecture
    ENC_DIM = 128  # encoder GRU hidden state size
    IC_DIM = 32  # Number of variables in 'initial conditions' AKA bottleneck
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
    P_DROPOUT = 0.95  # Proportion of units to set to 0 on each step. Unlike Tensorflow, JAX Dropout uses a 'keep' rate.
    L2_REG = 2.0e-5  # Parameter regularization strength.

    # Numerical stability HPs
    MAX_GRAD_NORM = 10.0  # gradient clipping above this value

    kl_warmup_start = 50.0  # batch number to start kl warmup, explicitly float
    kl_warmup_end = 250.0  # batch number to be finished with kl warmup, explicitly float
    kl_min = 0.01  # The minimum KL value, non-zero to make sure KL doesn't grow crazy before kicking in.
    kl_max = 1.0

    # Get the data #

    datadir = Path.cwd() / 'data' / 'joeyo'
    rng = random.PRNGKey(onp.random.randint(0, MAX_SEED_INT))
    X, Y, X_ax_info, Y_ax_info, targ_times = load_dat_with_vel_accel(datadir, SESS_IDX, trial_dur=MAX_TRIAL_DUR)
    trial_X, trial_tvec, true_bin_dur = bin_and_segment_spike_times(X, X_ax_info, targ_times,
                                                                    nearest_bin_dur=BIN_DURATION,
                                                                    trial_dur=MAX_TRIAL_DUR)
    n_trials, n_timesteps, n_neurons = trial_X.shape
    X_train, X_valid = train_test_split(trial_X, train_size=P_TRAIN)

    # Get the model #
    encoder_init, encode = LFADSEncoderModel(P_DROPOUT, ENC_DIM, IC_DIM)
    decoder_init, decode = LFADSDecoderModel(VAR_MIN, P_DROPOUT, GEN_DIM, n_timesteps, FACTORS_DIM, n_neurons)

    # Init the model
    rng, keys = utils.keygen(rng, 2)
    latent_shape, init_encoder_params = encoder_init(next(keys), (BATCH_SIZE, n_timesteps, n_neurons))
    decoded_shape, init_decoder_params = decoder_init(next(keys), (BATCH_SIZE, 0))

    # loglikelihood_t = Poisson(log_rates_t).logp(data_t_bxd)
    # num_batches = int(n_trials * EPOCHS / BATCH_SIZE)  # how many batches do we train

    decay_fun = optimizers.exponential_decay(STEP_SIZE, DECAY_STEPS, DECAY_FACTOR)

    opt_init, opt_update, get_params = optimizers.adam(step_size=decay_fun,
                                                       b1=0.9,
                                                       b2=0.999,
                                                       eps=1e-1)  # Seems small

    def binarize_batch(rng, i, images):
        i = i % num_batches
        batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
        return random.bernoulli(rng, batch)

    @jit
    def run_epoch(rng, opt_state):
        def body_fun(i, opt_state):
            elbo_rng, data_rng = random.split(random.fold_in(rng, i))
            batch = binarize_batch(data_rng, i, train_images)
            loss = lambda params: -elbo(elbo_rng, params, batch) / batch_size
            g = grad(loss)(get_params(opt_state))
            return opt_update(i, g, opt_state)

        def update_w_gc(i, opt_state, lfads_hps, lfads_opt_hps, key, x_bxt,
                        kl_warmup):
            """Update fun for gradients, includes gradient clipping."""
            params = get_params(opt_state)
            grads = grad(lfads.lfads_training_loss)(params, lfads_hps, key, x_bxt,
                                                    kl_warmup,
                                                    lfads_opt_hps['keep_rate'])
            clipped_grads = optimizers.clip_grads(grads, lfads_opt_hps['max_grad_norm'])
            return opt_update(i, clipped_grads, opt_state)

        return lax.fori_loop(0, num_batches, body_fun, opt_state)

    @jit
    def evaluate(opt_state, images):
        params = get_params(opt_state)
        elbo_rng, data_rng, image_rng = random.split(test_rng, 3)
        binarized_test = random.bernoulli(data_rng, images)
        test_elbo = elbo(elbo_rng, params, binarized_test) / images.shape[0]
        sampled_images = image_sample(image_rng, params, nrow, ncol)
        return test_elbo, sampled_images


    opt_state = opt_init(init_params)
    for epoch in range(num_epochs):
        tic = time.time()
        opt_state = run_epoch(random.PRNGKey(epoch), opt_state)
        test_elbo, sampled_images = evaluate(opt_state, test_images)
        print("{: 3d} {} ({:.3f} sec)".format(epoch, test_elbo, time.time() - tic))
        plt.imsave(imfile.format(epoch), sampled_images, cmap=plt.cm.gray)


