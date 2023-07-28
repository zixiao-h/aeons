from numpy import pi, log, exp
import numpy as np
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from aeons.utils import generate_Xs, X_mu, logXf_formula, calc_endpoints, reject_outliers
from aeons.regress import analytic_lm_params
from aeons.endpoint import theta_bandwidth_trunc

#| Define a four-dimensional spherical gaussian likelihood,
#| width sigma=0.1, centered on the 0 with one derived parameter.
#| The derived parameter is the squared radius

nDims = 10
nDerived = 0
sigma = 0.01

def likelihood(theta):
    """ Simple Gaussian Likelihood"""
    nDims = len(theta)
    r2 = sum(theta**2)
    logL = -log(2*pi*sigma*sigma)*nDims/2.0
    logL += -r2/2/sigma/sigma

    return logL, [r2]

#| Define a box uniform prior from -1 to 1

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(-1, 1)(hypercube)

#| Optional dumper function giving run-time read access to
#| the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    ndead = len(dead)
    nlive = len(live)
    if nlive == 0:
        print(f"[{'='*50}>]", end='\n')
        return
    logL = live[:, -1]
    # Sort logL in ascending order
    logL = np.sort(logL)
    nk = np.concatenate([nlive * np.ones(ndead), np.arange(nlive, 0, -1)])
    X_mean = X_mu(nk)[ndead:]
    # theta = analytic_lm_params(logL, X_mean, d0=2)
    # theta = theta_bandwidth_trunc(logL, X_mean, trunc=5, splits=[1])
    Nset = 25
    logXfs = np.zeros(Nset)
    for i in range(Nset):
        X = generate_Xs(nk)[ndead:]
        theta = analytic_lm_params(logL, X, d0=2)
        logXfs[i] = logXf_formula(theta, logZ, X_mean[0])
    logXfs = logXfs[~np.isnan(logXfs)]
    logXfs = reject_outliers(logXfs)
    endpoint, endpoint_std = calc_endpoints(len(dead), np.log(X_mean[0]), logXfs.mean(), logXfs.std(), nlive)
    end_predict = int(endpoint) + nlive
    # Print graphic showing progress of the run
    print(f"Predicted endpoint: {end_predict} +/- {int(endpoint_std)}, progress {len(dead)/end_predict*100:.0f}%")
    print(f"[{'='*int(len(dead)/end_predict*50)}>{'#'*(50-int(len(dead)/end_predict*50))}]", end='\n')

#| Initialise the settings

settings = PolyChordSettings(nDims, nDerived)
settings.base_dir = './chains'
settings.file_root = 'gaussian'
settings.nlive = 500
settings.do_clustering = False
settings.read_resume = False
settings.write_resume = False
settings.write_prior = False
settings.compression_factor = exp(-2)

#| Run PolyChord

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
