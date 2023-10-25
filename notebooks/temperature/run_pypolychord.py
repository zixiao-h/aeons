from numpy import pi, log
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from scipy.special import logsumexp

#| Define a four-dimensional spherical gaussian likelihood,
#| width sigma=0.1, centered on the 0 with one derived parameter.
#| The derived parameter is the squared radius

nDims = 8
nDerived = 0
sigma1 = 0.1
sigma2 = 0.01

def likelihood(theta):
    """ Simple Gaussian Likelihood"""

    nDims = len(theta)
    r2 = sum(theta**2)
    logL1 = -log(2*pi*sigma1*sigma1)*nDims/2.0
    logL1 += -r2/2/sigma1/sigma1

    logL2 = -log(2*pi*sigma2*sigma2)*nDims/2.0
    logL2 += -r2/2/sigma2/sigma2
    logL = logsumexp([logL1, logL2])

    return logL, []

#| Define a box uniform prior from -1 to 1

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(-1, 1)(hypercube)

#| Optional dumper function giving run-time read access to
#| the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    print(logZ)
    print("Last dead point:", dead[-1])

#| Initialise the settings

settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'slab_spike'
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False
settings.precision_criterion = 1e-4

#| Run PolyChord

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
