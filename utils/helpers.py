import numpy as np


def sample_normal(params, seed=None, best_fit=False):
    """
    Quick sample function for sampling parameters with normal distributions

    :param params: dictionary with either a single value, or a tuple of values structured as:
                    (best_estimate (center of distribution), stddev (width of distribution))
    :param seed: seed for random number generator
    :param best_fit: whether to sample strictly best fit parameter
    :return: dictionary of sampled parameters
    """
    # set random seed and init empty dict
    rng = np.random.default_rng(seed)
    sampled = {}
    for key, val in params.items():
        # if tuple of len 2, assume idx 0 is center and idx 1 is stddev, else set value
        if isinstance(val, tuple) and len(val) == 2:
            mean, sigma = val
            # if wanting only best fit values, else sample from normal distr
            if best_fit:
                sampled[key] = mean
            else:
                sampled[key] = rng.normal(loc=mean, scale=sigma)
        else:
            sampled[key] = val
    return sampled
