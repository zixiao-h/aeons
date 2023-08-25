from numpy import pi, log
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import GaussianPrior
from aeons.utils import *
from aeons.endpoint import EndModel
from aeons.regress import params_from_d
from aeons.likelihoods import full

try:
    from mpi4py import MPI
except ImportError:
    pass


#| Define a four-dimensional spherical gaussian likelihood,
#| width sigma=0.1, centered on the 0 with one derived parameter.
#| The derived parameter is the squared radius

nDims = 1
nDerived = 0

name, samples = get_samples('toy', 'gauss_16')
model = EndModel(samples)
ndead = 10000
points = points_at_iteration(samples, ndead)
logL, X_mean, nk, logZdead = model.data(ndead)
logLd, Xd = logL[ndead:], X_mean[ndead:]
N = len(logLd)
beta = get_beta(points, ndead)
dG = points.set_beta(beta).d_G(25).values
dG_mean = dG.mean()
dG_std = dG.std()


def likelihood(d):
    """ Simple Gaussian Likelihood"""
    theta = params_from_d(logLd, Xd, d)
    loss = logLd - full.func(Xd, theta)
    L_sq = np.sum(loss**2)
    s = np.sqrt(L_sq/N)
    logPr = -1/2 * N * log(2*pi*s**2) - L_sq/(2*s**2)
    return logPr, []

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return GaussianPrior(dG_mean, dG_std)(hypercube)

#| Optional dumper function giving run-time read access to
#| the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    print(logZ)
    print("Last dead point:", dead[-1])

#| Initialise the settings

settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'gaussian'
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False

#| Run PolyChord

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
