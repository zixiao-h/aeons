import numpy as np
import matplotlib.pyplot as plt

from aeons.utils import *
from aeons.endpoint import EndModel, theta_basic, theta_bandwidth, theta_bandwidth_trunc
from aeons.plotting import *
from aeons.regress import *

def logcauchy(X, theta, torched=False):
    logLmax, c, gamma = theta
    if torched:
        import torch
        return logLmax - (1 + c)/2 * torch.log( 1 + X**(2/c)/(gamma**2) )
    return logLmax - (1 + c)/2 * np.log( 1 + X**(2/c)/gamma**2 )

def logLmax_cauchy(logLi, Xi, c, gamma):
    N = len(logLi)
    summand = logLi + (1 + c)/2 * np.log( 1 + Xi**(2/c)/gamma**2 )
    return np.sum(summand)/N

# Load data
name, samples = get_samples("toy", "cauchy_10_0001")
model = EndModel(samples)
true_endpoint = model.true_endpoint()
true_logXf = samples.logX().iloc[true_endpoint]

ndead = 5000
logLall, Xall, nk, logZdead = model.data(ndead, live=False)
logLall, Xall = np.array(logLall), np.array(Xall)
window = np.arange(ndead, len(logLall))
logLd, Xd = logLall[window], Xall[window]


import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior

def likelihood(theta):
    c, gamma = theta
    logLmax = logLmax_cauchy(logLd, Xd, c, gamma)
    Lsq = np.sum((logLd - logcauchy(Xd, [logLmax, c, gamma]))**2)
    s = np.sqrt(Lsq/len(logLd))
    return -1/2 * len(logLd) * np.log(2*np.pi*s**2) - Lsq/(2*s**2), []

def prior(hypercube):
    c = UniformPrior(0, 1e3)(hypercube[0])
    gamma = LogUniformPrior(1e-20, 1)(hypercube[1])
    return np.array([c, gamma])

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

nDims = 2
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'cauchy_5000'
settings.nlive = 200
settings.do_clustering = False
settings.read_resume = False

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
paramnames = [('c', 'c'), ('gamma', r'\gamma')]
output.make_paramnames_files(paramnames)
