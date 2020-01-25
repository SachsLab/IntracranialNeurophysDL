import jax.numpy as np


def get_kl_warmup_fun(hps):
    """Warmup KL cost to avoid a pathological condition early in training.

    Arguments:
    hps : dictionary of optimization hyperparameters

    Returns:
    a function which yields the warmup value
    """
    def kl_warmup(batch_idx):
        progress_frac = ((batch_idx - hps['kl_warmup_start']) /
                         (hps['kl_warmup_end'] - hps['kl_warmup_start']))
        kl_warmup = np.where(batch_idx < hps['kl_warmup_start'], hps['kl_min'],
                             (hps['kl_max'] - hps['kl_min']) * progress_frac + hps['kl_min'])
        return np.where(batch_idx > hps['kl_warmup_end'], hps['kl_max'], kl_warmup)
    return kl_warmup
