import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gammainc, gamma, logsumexp, gammaincinv, loggamma

from aeons.regress import analytic_lm_params, GaussianRegress, params_from_d
from aeons.utils import *
from aeons.likelihoods import full

class EndModel:
    def __init__(self, samples):
        samples['logZs'] = np.logaddexp.accumulate(samples.logw())
        self.samples = samples
        self.logX_mean = np.array(samples.logX())
        self.logL = np.array(samples.logL)
        self.L = np.exp(self.logL - self.logL.max())
        self.logZs = samples['logZs']

    def true_endpoint(self, epsilon=1e-3):
        return calc_true_endpoint(self.logZs, epsilon)
    
    def true_logXf(self, epsilon=1e-3):
        true_endpoint = self.true_endpoint(epsilon)
        return self.logX_mean[true_endpoint]

    def points(self, ndead):
        return points_at_iteration(self.samples, ndead)

    def data(self, ndead, live=False):
        points = points_at_iteration(self.samples, ndead)
        logL = np.array(points.logL)
        nk = np.array(points.nlive)
        X_mean = X_mu(nk)
        logZdead = self.logZs.iloc[ndead]
        if live:
            return logL[ndead:], X_mean[ndead:], nk, logZdead
        return logL, X_mean, nk, logZdead

    def logXfs(self, method, iterations, **kwargs):
        """Method takes in arguments (ndead, logL, nk, logZdead, **kwargs) and returns logXfs_set"""
        clear = kwargs.pop('clear', True)
        N = len(iterations)
        logXfs = np.zeros(N)
        logXfs_std = np.zeros(N)
        for i, ndead in enumerate(iterations):
            points = self.points(ndead)
            try:
                logXfs_set, message = method(points, ndead, **kwargs)
                logXfs[i] = np.mean(logXfs_set)
                logXfs_std[i] = np.std(logXfs_set)
                if clear:
                    print('\r', f"Iteration {ndead} of {iterations[-1]}, {message}", end='')
                else:
                    print(f"Iteration {ndead} of {iterations[-1]}, {message}")
            except:
                logXfs[i] = np.nan
                logXfs_std[i] = np.nan
                print(f"Iteration {ndead} of {iterations[-1]}, NaN")
        return logXfs, logXfs_std


def logXf_basic(points, ndead, Nset=25):
    logL, X_mean, nk, logZdead = data(points)
    logLd = logL[ndead:]
    logXf_set = np.zeros(Nset)
    for i in range(Nset):
        X = generate_Xs(nk)
        Xd = X[ndead:]
        theta = analytic_lm_params(logLd, Xd, d0=1)
        logXf_set[i] = logXf_formula(theta, logZdead, X_mean[ndead])
    logXf_set = logXf_set[~np.isnan(logXf_set)]
    logXf_set = reject_outliers(logXf_set)
    return logXf_set, f'{len(logXf_set)} samples'


def logXf_beta_DKL(points, ndead, Nset=25):
    logL, X_mean, nk, logZdead = data(points)
    beta_DKL = get_beta(points, ndead)
    dG = points.set_beta(beta_DKL).d_G(Nset).values
    logLd = logL[ndead:]
    logXf_set = np.zeros(Nset)
    for i in range(Nset):
        X = generate_Xs(nk)
        Xd = X[ndead:]
        d = np.random.choice(dG)
        theta = params_from_d(logLd, Xd, d)
        logXf_set[i] = logXf_formula(theta, logZdead, X_mean[ndead])
    logXf_set = logXf_set[~np.isnan(logXf_set)]
    # logXf_set = reject_outliers(logXf_set)
    return logXf_set, f"{len(logXf_set)} samples, {dG.mean():.1f}"
