import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.regress import GaussianRegress
from aeons.likelihoods import full


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

def plot_logLx(samples, ndead=0, lives=True, ax=None, N_points=50, ms=2):
    points = points_at_iteration(samples, ndead)
    nk, logL = np.array(points.nlive), np.array(points.logL)
    X_mean = X_mu(nk)
    N = len(points)
    if lives:
        index = np.linspace(ndead, N, N_points, endpoint=False).astype(int)
    else:
        index = np.linspace(0, N, N_points, endpoint=False).astype(int)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,2))
        ax.plot(X_mean[index], logL[index], '+', ms=ms, color='navy')
        return fig, ax
    else:
        ax.plot(X_mean[index], logL[index], '+', ms=ms, color='navy')
    ax.ticklabel_format(style='sci')

def plot_std(xvals, y_means, y_stds, true, ylim=None, ax=None):
    # plt.plot(xvals, y_means, lw=1, color='navy')
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,2))
    ax.fill_between(xvals, y_means - y_stds, y_means + y_stds, alpha=1, color='deepskyblue')
    ax.fill_between(xvals, y_means - 2*y_stds, y_means + 2*y_stds, alpha=.2, color='deepskyblue')
    ax.axhline(y=true, lw=1, color='navy')
    if isinstance(ylim, float):
        ax.set_ylim(0, true*ylim)
    elif isinstance(ylim, tuple):
        ax.set_ylim(*ylim)

    
def plot_split(model, ndead, splits=2, trunc=None):
    logLlive, Xlive, nk, logZdead = model.data(ndead, live=True)
    regress = GaussianRegress(logLlive, Xlive)
    logXf = logXf_formula(regress.theta, logZdead, Xlive[0])
    endpoint = calc_endpoints(ndead, np.log(Xlive[0]), logXf, 0.01, nlive=750)[0]

    logL_split, X_split = data_split(logLlive, Xlive, splits, trunc)
    regress_split = GaussianRegress(logL_split, X_split)
    logXf_split = logXf_formula(regress_split.theta, logZdead, Xlive[0])
    endpoint_split = calc_endpoints(ndead, np.log(Xlive[0]), logXf_split, 0.01, nlive=750)[0]

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot(Xlive, logLlive, '+', ms=3)
    axs[0].plot(Xlive, full.func(Xlive, regress.theta), lw=2, label=f"{endpoint:.0f}")
    axs[0].set_title(f"{formatt(regress.theta)} logZ = {regress.logZ():.1f}")
    axs[0].legend()
    axs[1].plot(Xlive, logLlive, '+', ms=3)
    axs[1].plot(X_split, full.func(X_split, regress_split.theta), lw=2, label=f"{endpoint_split:.0f}")
    axs[1].set_title(f"{formatt(regress_split.theta)} logZ = {regress_split.logZ():.1f}")
    axs[1].set_yticks([])
    axs[1].legend()
    fig.suptitle(f"ndead = {ndead}, true = {model.true_endpoint():.0f}, {model.true_logXf():.1f}", y=0.95)
    fig.tight_layout()
 