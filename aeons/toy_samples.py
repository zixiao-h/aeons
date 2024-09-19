import numpy as np
from anesthetic.examples.utils import (
    random_ellipsoid, random_covariance, volume_n_ball, log_volume_n_ball)
from anesthetic.examples.perfect_ns import (
    gaussian, correlated_gaussian, wedding_cake)
from anesthetic import NestedSamples
from anesthetic.samples import merge_nested_samples

def correlated_gaussian_samples(nlive=500, ndims=5, sigma=0.1):
    np.random.seed(0)
    mean = 0.5*np.ones(ndims)
    cov = random_covariance(np.random.rand(ndims)*np.sqrt(2)*sigma)
    samples = correlated_gaussian(nlive, mean, cov)
    return samples

def gaussian_samples(nlive=500, ndims=10, sigma=0.1, seed=0):
    np.random.seed(seed)
    R = 1
    samples = gaussian(nlive, ndims, sigma, R)
    return samples

def wedding_cake_samples(nlive=500, ndims=4, sigma=0.01, alpha=0.5):
    np.random.seed(0)
    samples = wedding_cake(nlive, ndims, sigma, alpha)
    return samples

def cauchy_samples(nlive, ndims, gamma=0.1, R=1, logLmin=-1e-2):
    def logLike(x):
        return - (1+ndims)/2  *np.log( 1 + (x**2).sum(axis=-1)/(gamma**2) )
        
    def random_sphere(n):
        return random_ellipsoid(np.zeros(ndims), np.eye(ndims), n)

    samples = []
    r = R
    logL_birth = np.ones(nlive) * -np.inf
    logL = logL_birth.copy()
    while logL.min() < logLmin:
        points = r * random_sphere(nlive)
        logL = logLike(points)
        samples.append(NestedSamples(points, logL=logL, logL_birth=logL_birth))
        logL_birth = logL.copy()
        r = (points**2).sum(axis=-1, keepdims=True)**0.5

    samples = merge_nested_samples(samples)
    samples.logL
    logLend = samples[samples.nlive >= nlive].logL.max()
    return samples[samples.logL_birth < logLend].recompute()