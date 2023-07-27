import numpy as np
import matplotlib.pyplot as plt

from aeons.utils import *
from aeons.endpoint import EndModel, theta_basic, theta_bandwidth, theta_bandwidth_trunc
from aeons.plotting import *
from aeons.regress import *

def logf4(X, theta):
    logLmax, Delta, d, gamma = theta
    return logLmax - Delta * np.log(1 + X**(2/d)/gamma**2)

def logLmax_Delta_estimator(logL, X, d, gamma):
    N = len(logL)
    logXsum = np.sum( np.log(1 + X**(2/d)/gamma**2) )
    logLlogXsum = np.sum( logL * np.log(1 + X**(2/d)/gamma**2) )
    logXlogXsum = np.sum( (np.log(1 + X**(2/d)/gamma**2))**2 )
    logLsum = np.sum(logL)
    denom = (logXsum**2)/N - logXlogXsum
    logLmax_estimator = (logLlogXsum * logXsum/N - logXlogXsum * logLsum/N)/denom
    Delta_estimator = (logLlogXsum - logXsum * logLsum/N)/denom
    return logLmax_estimator, Delta_estimator

# Load data
name, samples = get_samples("toy", "gauss_30_01")
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
    d, gamma = theta
    logLmax, Delta = logLmax_Delta_estimator(logLd, Xd, d, gamma)
    Lsq = np.sum((logLd - logf4(Xd, [logLmax, Delta, d, gamma]))**2)
    s = np.sqrt(Lsq/len(logLd))
    return -1/2 * len(logLd) * np.log(2*np.pi*s**2) - Lsq/(2*s**2), []

def prior(hypercube):
    c = UniformPrior(0, 1e3)(hypercube[0])
    gamma = LogUniformPrior(1e-10, 1)(hypercube[1])
    return np.array([c, gamma])

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

nDims = 2
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'gauss_5000'
settings.nlive = 100
settings.do_clustering = False
settings.read_resume = False

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
paramnames = [('c', 'c'), ('gamma', r'\gamma')]
output.make_paramnames_files(paramnames)
