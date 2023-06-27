import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gamma, logsumexp, gammaincinv, loggamma

from aeons.lm_partial import analytic_lm_params
from aeons.covariance import points_at_iteration, X_mu
from aeons.true_distribution import generate_Xs
from aeons.likelihoods import full
from aeons.models import LS

def reject_outliers(data):
    data = np.array(data)
    dev = np.abs(data - np.median(data))
    median_dev = np.median(dev)
    return data[dev < 2 * median_dev]

def logXf_formula(theta, logZdead, Xi, epsilon=1e-3):
    logLmax, d, sigma = theta
    loglive = np.log( gamma(d/2) * gammainc(d/2, Xi**(2/d)/(2*sigma**2)) )
    logdead = logZdead - logLmax - (d/2)*np.log(2) - d*np.log(sigma) + np.log(2/d)
    logend = logsumexp([loglive, logdead]) + np.log(epsilon)
    xf_reg = gammaincinv(d/2, np.exp(logend)/gamma(d/2))
    return d/2 * np.log(2*sigma**2 * xf_reg)

def logXf_formula_gaussian(theta, epsilon=1e-3):
    _, d, sigma = theta
    return d/2*np.log(2) + d * np.log(sigma) + loggamma(1 + d/2) + np.log(epsilon)


def end_iteration(samples, N, start=0.5, epsilon=1e-3):
    iterations = np.linspace(len(samples) * start, len(samples), N, endpoint=False).astype(int)
    for ndead in iterations:
        points = points_at_iteration(samples, ndead)
        weights = points.logw()
        if logsumexp(weights[ndead:]) - logsumexp(weights) < np.log(epsilon):
            return ndead
        
def minimise_bandwidth(logL, X_mean, ndead, alphas, x0, warnings=False, give_alpha=False):
    theta_best = analytic_lm_params(logL[ndead:], X_mean[ndead:], x0)
    alpha_best = 0
    logZi = -np.inf
    for alpha in alphas:
        startf = int(ndead * (1 - alpha))
        logLf = logL[startf:]
        X_meanf = X_mean[startf:]
        ls = LS(logLf, full, X_meanf)
        theta = analytic_lm_params(logLf, X_meanf, x0)
        try:
            logZf = ls.logZ(theta) - startf * np.log(startf) + startf
        except RuntimeWarning:
            if warnings:
                print(np.round(theta, 2), 'warning', alpha)
            continue
        if logZf > logZi:
            logZi = logZf
            theta_best = theta
            alpha_best = alpha
    if give_alpha:
        return theta_best, alpha_best
    return theta_best

class EndModel:
    def __init__(self, samples):
        self.samples = samples
        self.logX_mean = np.array(samples.logX())
        self.logL = np.array(samples.logL)
        self.L = np.exp(self.logL)
        self.nlive = float(samples.nlive[0])
    
    def logXf_true(self, N=50, start=0.5):
        kf = end_iteration(self.samples, N, start)
        return self.logX_mean[kf]

    def points(self, ndead):
        return points_at_iteration(self.samples, ndead)
    
    def data(self, ndead):
        points = points_at_iteration(self.samples, ndead)
        logL = np.array(points.logL)
        nk = np.array(points.nlive)
        logZdead = logsumexp(points.logw()[:ndead])
        return logL, nk, logZdead
    
    def minimise(self, ndead, Nset=None):
        logL, nk, _ = self.data(ndead)
        if Nset:
            theta = []
            for _ in range(Nset):
                X = generate_Xs(nk)
                theta.append(analytic_lm_params(logL[ndead:], X[ndead:], 1))
            return theta
        else:
            return analytic_lm_params(logL[ndead:], X_mu(nk)[ndead:], 1)
        
    def minimise_bandwidth(self, ndead, alphas, Nset=None, give_alpha=False):
        logL, nk, _ = self.data(ndead)
        X_mean = X_mu(nk)
        theta_best, alpha_best = minimise_bandwidth(logL, X_mean, ndead, alphas, x0=1, give_alpha=True)
        if Nset:
            theta = []
            for _ in range(Nset):
                X = generate_Xs(nk)
                try:
                    startf = int((1 - alpha_best) * ndead)
                    theta_l = analytic_lm_params(logL[startf:], X[startf:], 1)
                    theta.append(theta_l)
                except RuntimeWarning:
                    continue
            if give_alpha:
                return theta, alpha_best
            else:
                return theta
        else:
            if give_alpha:
                return theta_best, alpha_best
            else:
                return theta_best

    
    def logXfs_mean_gaussian(self, iterations, epsilon=1e-3):
        N = len(iterations)
        logXfs = np.zeros(N)
        for i, ndead in enumerate(iterations):
            logL, nk = self.data(ndead)
            X_mean = X_mu(nk)
            theta = analytic_lm_params(logL[ndead:], X_mean[ndead:], 10)
            logXfs[i] = logXf_formula_gaussian(theta, epsilon)
        return logXfs

    def logXfs(self, iterations, Nset=None, epsilon=1e-3):
        N = len(iterations)
        if Nset:
            logXfs = np.zeros(N)
            logXfs_std = np.zeros(N)
            for i, ndead in enumerate(iterations):
                logL, nk, logZdead = self.data(ndead)
                logXf_i = np.zeros(Nset)
                for j in range(Nset):
                    X = generate_Xs(nk)
                    theta = analytic_lm_params(logL[ndead:], X[ndead:], 1)
                    logXf_i[j] = logXf_formula(theta, logZdead, X[ndead], epsilon)
                logXf_i = logXf_i[~np.isnan(logXf_i)]
                logXf_i = reject_outliers(logXf_i)
                logXfs[i] = np.mean(logXf_i)
                logXfs_std[i] = np.std(logXf_i)
            return logXfs, logXfs_std
        else:
            logXfs = np.zeros(N)
            for i, ndead in enumerate(iterations):
                logL, nk, logZdead = self.data(ndead)
                X_mean = X_mu(nk)
                theta = analytic_lm_params(logL[ndead:], X_mean[ndead:], 1)
                logXfs[i] = logXf_formula(theta, logZdead, X_mean[ndead], epsilon)
            return logXfs
        
    def logXfs_bandwidth(self, iterations, alphas, Nset=None, epsilon=1e-3):
        N = len(iterations)
        if Nset:
            logXfs = np.zeros(N)
            logXfs_std = np.zeros(N)
            for i, ndead in enumerate(iterations):
                logL, nk, logZdead = self.data(ndead)
                X_mean = X_mu(nk)
                logXf_i = np.zeros(Nset)
                theta_best, alpha_best = minimise_bandwidth(logL, X_mean, ndead, alphas, x0=1, give_alpha=True)
                for j in range(Nset):
                    X = generate_Xs(nk)
                    startf = int((1 - alpha_best) * ndead)
                    theta = analytic_lm_params(logL[startf:], X[startf:], 1)
                    logXf_i[j] = logXf_formula(theta, logZdead, X[ndead], epsilon)
                logXf_i = logXf_i[~np.isnan(logXf_i)]
                logXf_i = reject_outliers(logXf_i)
                logXfs[i] = np.mean(logXf_i)
                logXfs_std[i] = np.std(logXf_i)
            return logXfs, logXfs_std
        else:
            logXfs = np.zeros(N)
            for i, ndead in enumerate(iterations):
                logL, nk, logZdead = self.data(ndead)
                X_mean = X_mu(nk)
                theta_best = minimise_bandwidth(logL, X_mean, ndead, alphas, 1)
                logXfs[i] = logXf_formula(theta_best, logZdead, X_mean[ndead], epsilon)
            return logXfs

    def plot_theta(self, theta):
        logXarray = np.flip(np.linspace(self.logX_mean.min(), 0, 2000))
        Xarray = np.exp(logXarray)
        plt.plot(self.logX_mean, self.L, lw=1, color='black')
        plt.plot(logXarray, np.exp(full.func(Xarray, theta)), lw=1, color='orange')

    def plot_lx(self):
        fig, ax1 = plt.subplots(figsize=(6.7,2))
        ax2 = plt.twinx(ax1)
        ax1.plot(self.logX_mean, self.L)
        ax2.plot(self.logX_mean, self.L*np.exp(self.logX_mean))
