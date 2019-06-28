"""
Copied (and slightly modified) from OpenAI Baselines
"""
import numpy as np


def apply_normalizer(data, normalizer, update_data=None, center=True,
                     clip_limit=10):
    """Apply a RunningMeanStd normalizer to an array."""
    if update_data is not None:
        # Update the statistics with different data than we're normalizing
        normalizer.update(update_data.reshape((-1, ) + normalizer.shape))
    else:
        normalizer.update(data.reshape((-1, ) + normalizer.shape))
    if center:
        data = data - normalizer.mean
    data = data / np.sqrt(normalizer.var + 1e-8)
    data = np.clip(data, -clip_limit, clip_limit)
    return data

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.shape = shape

    def update(self, x):
        """x must have shape (-1, self.shape[0], self.shape[1], etc)"""
        assert x.shape[1:] == self.shape, (x.shape, self.shape)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
