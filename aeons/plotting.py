import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import generate_Xs, points_at_iteration


def plot_lx(samples):
    logL = samples.logL
    logX_mean = samples.logX()
    fig, ax1 = plt.subplots(figsize=(6.7,2))
    ax2 = plt.twinx(ax1)
    logL_norm = logL - logL.max()
    L_norm = np.exp(logL_norm)
    ax1.plot(logX_mean, L_norm, lw=1, color='black')
    ax2.plot(logX_mean, L_norm*np.exp(logX_mean), lw=1, color='navy')
    ax1.set_title(f'logL_max = {logL.max():.2f}')

def plot_logLx(samples, ndead=None, Nset=None):
    logL = samples.logL
    logX_mean = samples.logX()

def plot_endpoints(iterations, endpoints, endpoints_std, true_endpoint, ylim=1.1, logX_vals=None):
    # plt.plot(iterations, endpoints, lw=1, color='navy')
    if logX_vals is not None:
        iterations = -logX_vals
    plt.fill_between(iterations, endpoints - endpoints_std, endpoints + endpoints_std, alpha=1, color='deepskyblue')
    plt.fill_between(iterations, endpoints - 2*endpoints_std, endpoints + 2*endpoints_std, alpha=.2, color='deepskyblue')
    plt.axhline(y=true_endpoint, lw=1, color='navy')
    plt.ylim(0, true_endpoint*ylim)

