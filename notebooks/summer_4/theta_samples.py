import numpy as np
import matplotlib.pyplot as plt
from cauchy_regress import logcauchy, logLmax_cauchy

import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, LogUniformPrior

def call_polychord(logL, X, file_root, nlive=50):
    def likelihood(theta):
        c, gamma = theta
        logLmax = logLmax_cauchy(logL, X, c, gamma)
        Lsq = np.sum((logL - logcauchy(X, [logLmax, c, gamma]))**2)
        s = np.sqrt(Lsq/len(logL))
        return -1/2 * len(logL) * np.log(2*np.pi*s**2) - Lsq/(2*s**2), []

    def prior(hypercube):
        c = UniformPrior(0, 1e3)(hypercube[0])
        gamma = LogUniformPrior(1e-20, 1)(hypercube[1])
        return np.array([c, gamma])

    def dumper(live, dead, logweights, logZ, logZerr):
        print("Last dead point:", dead[-1])

    nDims = 2
    nDerived = 0
    settings = PolyChordSettings(nDims, nDerived)
    settings.file_root = file_root
    settings.nlive = nlive
    settings.do_clustering = False
    settings.read_resume = False

    output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
    paramnames = [('c', 'c'), ('gamma', r'\gamma')]
    output.make_paramnames_files(paramnames)
    
    from anesthetic import read_chains
    samples = read_chains(settings.base_dir + '/' + settings.file_root)
    return samples

