import numpy as np
import matplotlib.pyplot as plt

from aeons.utils import *
from aeons.endpoint import EndModel
from aeons.toy_samples import gaussian_samples
from aeons.regress import analytic_lm_params, GaussianRegress
from aeons.likelihoods import full

samples = gaussian_samples(nlive=200, ndims=4, sigma=0.1, seed=20)
model = EndModel(samples)
ndead = 1200
logL, X_mean, nk, logZdead = model.data(ndead)
logX_mean = logX_mu(nk)

logXarray = np.linspace(0, model.logX_mean[-1], 1000)
Xarray = np.exp(logXarray)

figsettings()
fig, ax1 = plt.subplots(figsize=(4, 1.6))
Nset = 3
colors = ['navy', 'cornflowerblue', 'lightsteelblue']
for i in range(Nset):
    X = generate_Xs(nk)[ndead:]
    regress = GaussianRegress(logL[ndead:], X)
    logLmax, d, sigma = regress.theta
    covtheta = regress.covtheta()
    ax1.plot(-np.log(X), np.exp(logL[ndead:]), '+', ms=2, color=colors[i]) # samples to fit over
    for _ in range(10):
        t = np.random.multivariate_normal(regress.theta, covtheta)
        ax1.plot(-logXarray, np.exp(full.func(Xarray, t)), lw=.1, color=colors[i]) # reconstructed likelihood

ax1.plot([], [], '+', ms=1, color='navy', label='Live points')
ax1.plot([], [], lw=.5, color='deepskyblue', label='Least squares fits')
ax1.plot(-logXarray, np.exp(full.func(Xarray, [0, 4, 0.1])), lw=2, color='black', label='True likelihood') # true likelihood
ax1.set_xticks([]);
ax1.set_yticks([]);
# ax1.set_xlim(logX_mean[-1], logX_mean[ndead])
# ax1.set_ylim(np.exp(logL[ndead]), np.exp(logL[-1]))
ax1.set_xlabel('$-\log X$')
ax1.set_ylabel('$\mathcal{L}(X)$')
ax1.margins(x=0, y=0)
ax1.legend(fontsize=6)
plt.savefig('prediction_uncertainty.pdf', pad_inches=0, bbox_inches='tight')
