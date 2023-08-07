import numpy as np
import matplotlib.pyplot as plt

from aeons.utils import *
from aeons.likelihoods import full

theta = [0, 10, 0.01]
logX = np.linspace(-60, 0, 1000)
X = np.exp(logX)
logL = full.func(X, theta)
L = np.exp(logL - logL.max())
LX = L * X

ndead = 5000
nlive = 500

def gaussian(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)
ElogXlivemin = -ndead/nlive - np.log(nlive) - 0.577
StdlogXlivemin = np.sqrt(ndead/nlive**2 + (np.pi**2)/6)
live_logX = logX[(logX > ElogXlivemin) * (logX < -ndead/nlive)]

figsettings()
plt.figure(figsize=(4,1.5))
plt.plot(logX, L/L.max(), color='black', label='Likelihood', lw=.5)
plt.fill(logX, LX/LX.max(), alpha=0.4, color='gray', label='Posterior')
plt.fill_between(live_logX, np.zeros(len(live_logX)), np.ones_like(live_logX)/(np.log(nlive) + 0.577), alpha=0.5, color='deepskyblue', label='Live points')
plt.plot(logX, gaussian(logX, ElogXlivemin, StdlogXlivemin), lw=.5, color='red', label='Minimum live point')
plt.axvline(-ndead/nlive, color='deepskyblue', lw=.5, ls='--')
plt.xlabel(r'$\log X$')
plt.legend()
plt.savefig('live_point_distribution.pdf', bbox_inches='tight')
