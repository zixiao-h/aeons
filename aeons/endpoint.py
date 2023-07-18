import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gamma, logsumexp, gammaincinv, loggamma

from aeons.regress import analytic_lm_params, GaussianRegress
from aeons.utils import points_at_iteration, generate_Xs, reject_outliers, logXf_formula, add_logZ, calc_true_endpoint
from aeons.likelihoods import full


class EndModel:
    def __init__(self, samples):
        self.samples = add_logZ(samples)
        self.logX_mean = np.array(samples.logX())
        self.logL = np.array(samples.logL)
        self.L = np.exp(self.logL)
        self.logZs = samples['logZs']

    def true_endpoint(self, epsilon=1e-3):
        return calc_true_endpoint(self.logZs, epsilon)
    
    def true_logXf(self, epsilon=1e-3):
        true_endpoint = self.true_endpoint(epsilon)
        return self.logX_mean[true_endpoint]

    def data(self, ndead):
        points = points_at_iteration(self.samples, ndead)
        logL = np.array(points.logL)
        nk = np.array(points.nlive)
        logZdead = logsumexp(points.logw()[:ndead])
        return logL, nk, logZdead

    def logXfs(self, method, iterations, Nset=10, trunc=15, epsilon=1e-3, **kwargs):
        N = len(iterations)
        logXfs = np.zeros(N)
        logXfs_std = np.zeros(N)
        for i, ndead in enumerate(iterations):
            logL, nk, logZdead = self.data(ndead)
            logXf_i = np.zeros(Nset)
            for j in range(Nset):
                X = generate_Xs(nk)
                theta = method(logL[ndead:], X[ndead:], **kwargs)
                logXf_i[j] = logXf_formula(theta, logZdead, X[ndead], epsilon)
            logXf_i = logXf_i[~np.isnan(logXf_i)]
            logXf_i = reject_outliers(logXf_i)
            logXfs[i] = np.mean(logXf_i)
            logXfs_std[i] = np.std(logXf_i)
            print(f"Iteration {ndead} complete, {len(logXf_i)} samples")
        return logXfs, logXfs_std


def theta_basic(logL, X):
    return analytic_lm_params(logL, X, d0=1)

def theta_bandwidth(logL, X, print_split=False, splits=4):
    logZmax = -np.inf
    theta_best = None
    split_best = None
    if isinstance(splits, int):
        splits = np.arange(1, splits + 1)
    for split in splits:
        start = len(X) - int(len(X)/split)
        Xs, logLs = X[start:], logL[start:]
        regress = GaussianRegress(logLs, Xs)
        theta = regress.theta
        logZ = regress.logZ()
        if logZ > logZmax:
            logZmax = logZ
            theta_best = theta
            split_best = split
    if print_split:
        print(f"Best split: {split_best}, {theta_best}")
    return theta_best

def theta_bandwidth_trunc(logL, X, trunc=15, **kwargs):
    theta = theta_bandwidth(logL, X)
    attempts = 1
    while theta[1]/2 > 180:
        theta = theta_bandwidth(logL[:-attempts*trunc], X[:-attempts*trunc], **kwargs)
        attempts += 1
    if attempts > 1:
        print(f"{attempts} attempts")
    return theta

