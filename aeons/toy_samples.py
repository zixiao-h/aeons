import numpy as np
from anesthetic.examples.utils import (
    random_ellipsoid, random_covariance, volume_n_ball, log_volume_n_ball)
from anesthetic.examples.perfect_ns import (
    gaussian, correlated_gaussian, wedding_cake)

def correlated_gaussian_samples(nlive=500, ndims=5, sigma=0.1):
    np.random.seed(0)
    mean = 0.5*np.ones(ndims)
    cov = random_covariance(np.random.rand(ndims)*sigma)
    samples = correlated_gaussian(nlive, mean, cov)
    return samples

def gaussian_samples(nlive=500, ndims=10, sigma=0.1):
    np.random.seed(0)
    R = 1
    samples = gaussian(nlive, ndims, sigma, R)
    return samples

def wedding_cake_samples(nlive=500, ndims=4, sigma=0.01):
    np.random.seed(0)
    nlive = 500
    ndims = 4
    sigma = 0.01
    alpha = 0.5
    samples = wedding_cake(nlive, ndims, sigma, alpha)
    return samples