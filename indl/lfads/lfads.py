import jax.numpy as np
from jax import random
import indl.lfads.distributions as dists
import indl.lfads.utils as utils


def linear_params(key, o, u, ifactor=1.0):
    """
    Params for y = w x

    Arguments:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

    Returns:
    a dictionary of parameters
    """
    key, skeys = utils.keygen(key, 1)
    ifactor = ifactor / np.sqrt(u)
    return {'w' : random.normal(next(skeys), (o, u)) * ifactor}


def affine_params(key, o, u, ifactor=1.0):
    """
    Params for y = w x + b

    Arguments:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

    Returns:
    a dictionary of parameters
    """
    key, skeys = utils.keygen(key, 1)
    ifactor = ifactor / np.sqrt(u)
    return {'w': random.normal(next(skeys), (o, u)) * ifactor,
            'b': np.zeros((o,))}


def gru_params(key, n, u, ifactor=1.0, hfactor=1.0, hscale=0.0):
    """
    Generate GRU parameters

    Arguments:
    key: random.PRNGKey for random bits
    n: hidden state size
    u: input size
    ifactor: scaling factor for input weights
    hfactor: scaling factor for hidden -> hidden weights
    hscale: scale on h0 initial condition

    Returns:
    a dictionary of parameters
    """
    key, skeys = utils.keygen(key, 5)
    ifactor = ifactor / np.sqrt(u)
    hfactor = hfactor / np.sqrt(n)

    wRUH = random.normal(next(skeys), (n+n,n)) * hfactor
    wRUX = random.normal(next(skeys), (n+n,u)) * ifactor
    wRUHX = np.concatenate([wRUH, wRUX], axis=1)

    wCH = random.normal(next(skeys), (n,n)) * hfactor
    wCX = random.normal(next(skeys), (n,u)) * ifactor
    wCHX = np.concatenate([wCH, wCX], axis=1)

    return {'h0': random.normal(next(skeys), (n,)) * hscale,
            'wRUHX': wRUHX,
            'wCHX': wCHX,
            'bRU': np.zeros((n+n,)),
            'bC': np.zeros((n,))}


def lfads_params(key, lfads_hps):
    """Instantiate random LFADS parameters.

    Arguments:
    key: random.PRNGKey for random bits
    lfads_hps: a dict of LFADS hyperparameters

    Returns:
    a dictionary of LFADS parameters
    """
    key, skeys = utils.keygen(key, 10)

    data_dim = lfads_hps['data_dim']
    ntimesteps = lfads_hps['ntimesteps']
    enc_dim = lfads_hps['enc_dim']
    con_dim = lfads_hps['con_dim']
    ii_dim = lfads_hps['ii_dim']
    gen_dim = lfads_hps['gen_dim']
    factors_dim = lfads_hps['factors_dim']

    ic_enc_params = {'fwd_rnn' : gru_params(next(skeys), enc_dim, data_dim),
                     'bwd_rnn' : gru_params(next(skeys), enc_dim, data_dim)}
    gen_ic_params = affine_params(next(skeys), 2*gen_dim, 2*enc_dim)  # m,v <- bi
    ic_prior_params = dists.diagonal_gaussian_params(next(skeys), gen_dim, 0.0,
                                                   lfads_hps['ic_prior_var'])
    con_params = gru_params(next(skeys), con_dim, 2*enc_dim + factors_dim)
    con_out_params = affine_params(next(skeys), 2*ii_dim, con_dim)  # m,v
    ii_prior_params = dists.ar1_params(next(skeys), ii_dim,
                                       lfads_hps['ar_mean'],
                                       lfads_hps['ar_autocorrelation_tau'],
                                       lfads_hps['ar_noise_variance'])
    gen_params = gru_params(next(skeys), gen_dim, ii_dim)
    factors_params = linear_params(next(skeys), factors_dim, gen_dim)
    lograte_params = affine_params(next(skeys), data_dim, factors_dim)

    return {'ic_enc': ic_enc_params,
            'gen_ic': gen_ic_params,
            'ic_prior': ic_prior_params,
            'con': con_params,
            'con_out': con_out_params,
            'ii_prior': ii_prior_params,
            'gen': gen_params,
            'factors': factors_params,
            'logrates': lograte_params}
