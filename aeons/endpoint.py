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

    def d_Gs(self, method, iterations, **kwargs):
        N = len(iterations)
        d_Gs, d_Gs_std = np.zeros(N), np.zeros(N)
        for i, ndead in enumerate(iterations):
            points = self.points(ndead)
            try:
                d_Gs_set, message = method(points, ndead, **kwargs)
                d_Gs[i] = np.mean(d_Gs_set)
                d_Gs_std[i] = np.std(d_Gs_set)
                print('\r', f"Iteration {ndead} of {iterations[-1]}, {message}", end='')
            except:
                d_Gs[i] = np.nan
                d_Gs_std[i] = np.nan
                print(f"Iteration {ndead} of {iterations[-1]}, NaN")
        return d_Gs, d_Gs_std
    
    def inferences(self, d_G_method, iterations, Nset=25):
        logXfs, d_Gs = np.zeros((len(iterations), Nset)), np.zeros((len(iterations), Nset))
        for i, ndead in enumerate(iterations):
            points = self.points(ndead)
            d_G = d_G_method(points, ndead, Nset)[0]
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(d_G)
            logL, X_mean, nk, logZdead = data(points)
            logXf_set = np.zeros(Nset)
            ds = kde.resample()
            ds = ds[ds > 0]
            for j in range(Nset):
                X = generate_Xs(nk)
                d = np.random.choice(ds)
                theta = params_from_d(logL[ndead:], X[ndead:], d)
                logXf_set[j] = logXf_formula(theta, logZdead, X_mean[ndead])
            logXfs[i], d_Gs[i] = logXf_set, d_G
            print('\r', f"Iteration {ndead} of {iterations[-1]}, d={d_G.mean():.1f}", end='')
        return logXfs, d_Gs

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


def logXf_dG_DKL(points, ndead, Nset=25):
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

def get_beta_end(points, ndead, epsilon=1e-3):
    logX = points.logX()
    logL = points.logL
    # Check if already terminated
    logw = logL + logX
    w = np.exp(logw - logw.max())
    Zdead = np.sum(w[:ndead])
    Zlive = np.sum(w[ndead:])
    if (Zdead != 0):
        if (Zlive/Zdead < epsilon):
            return 1
    # Otherwise, find beta
    def func(beta):
        logw = beta * logL + logX
        w = np.exp(logw - logw.max())
        Zdead = np.sum(w[:ndead])
        Zlive = np.sum(w[ndead:])
        if (Zdead == 0):
            return np.inf
        return Zlive/Zdead - epsilon
    from scipy import optimize
    try:
        res = optimize.root_scalar(func, bracket=[0, 1])
        return res.root
    except:
        return 0
    
def get_beta_start(points, ndead, epsilon=1e-3):
    logX = points.logX()
    logL = points.logL
    # Check if already terminated
    logw = logL + logX
    w = np.exp(logw - logw.max())
    Zdead = np.sum(w[:ndead])
    Zlive = np.sum(w[ndead:])
    if Zlive/Zdead < epsilon:
        return 1
    # Otherwise, find beta
    def func(beta):
        logw = beta * logL + logX
        w = np.exp(logw - logw.max())
        Zdead = np.sum(w[:ndead])
        Zlive = np.sum(w[ndead:])
        return Zlive/Zdead - 1/epsilon
    from scipy import optimize
    try:
        res = optimize.root_scalar(func, bracket=[0, 1e6])
        return res.root
    except:
        return 0
    

def logXf_dG_range(points, ndead, Nset=25):
    logL, X_mean, nk, logZdead = data(points)
    beta_end = get_beta_end(points, ndead)
    beta_start = get_beta_start(points, ndead)
    if beta_end == 0:
        betas = np.linspace(0, beta_start, 10)
    elif (beta_end*beta_start == 1):
        betas = 1
    else:
        betas = np.exp(np.linspace(np.log(beta_end), np.log(beta_start), 10))
    dG = points.d_G(nsamples=25, beta=betas)
    dG_mean, dG_std = dG.mean(), dG.std()
    
    logXf_set = np.zeros(Nset)
    for i in range(Nset):
        X = generate_Xs(nk)
        d = np.random.normal(dG_mean, dG_std)
        while d < 0:
            d = np.random.normal(dG_mean, dG_std)
        theta = params_from_d(logL[ndead:], X[ndead:], d)
        logXf_set[i] = logXf_formula(theta, logZdead, X_mean[ndead])
    logXf_set = logXf_set[~np.isnan(logXf_set)]
    logXf_set = reject_outliers(logXf_set, 3)
    return logXf_set, f"{beta_start:.2e}, {beta_end:.2e}"
