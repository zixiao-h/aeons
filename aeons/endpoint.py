import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gammainc, gamma, logsumexp, gammaincinv, loggamma

from aeons.regress import analytic_lm_params, GaussianRegress
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

    def data(self, ndead, live=False):
        points = points_at_iteration(self.samples, ndead)
        logL = np.array(points.logL)
        nk = np.array(points.nlive)
        X_mean = X_mu(nk)
        logZdead = self.logZs.iloc[ndead]
        if live:
            return logL[ndead:], X_mean[ndead:], nk, logZdead
        return logL, X_mean, nk, logZdead

    def logXfs(self, method, iterations, Nset=10, trunc=15, epsilon=1e-3, **kwargs):
        N = len(iterations)
        logXfs = np.zeros(N)
        logXfs_std = np.zeros(N)
        for i, ndead in enumerate(iterations):
            logL, X_mean, nk, logZdead = self.data(ndead)
            logXf_i = np.zeros(Nset)
            for j in range(Nset):
                X = generate_Xs(nk)
                theta = method(logL[ndead:], X[ndead:], **kwargs)
                logXf_i[j] = logXf_formula(theta, logZdead, X_mean[ndead], epsilon)
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
    while (theta is None) or (theta[1]/2 > 180) or ((trunc * attempts) > len(logL)):
        try:
            theta = theta_bandwidth(logL[:-attempts*trunc], X[:-attempts*trunc], **kwargs)
            attempts += 1
        except:
            attempts +=1
    if attempts > 1:
        print(f"{attempts} attempts")
    return theta

def get_dlogZ(logZ):
    logZ = pd.Series(logZ)
    dlogZ = logZ.diff(1)[1:]
    return dlogZ

def get_dlogZ_rolling(dlogZ, N_rolling):
    dlogZ = pd.Series(dlogZ)
    dlogZ_rolling = dlogZ.rolling(N_rolling).mean()
    dlogZ_rolling.dropna(inplace=True)
    return dlogZ_rolling

import numpy.polynomial.polynomial as poly
def fit_dlogZ(dlogZ, deg):
    x = dlogZ.index.get_level_values(0).values
    y = dlogZ.values
    coefs = poly.polyfit(x, y, deg)
    return coefs


from aeons.endpoint import *
class IncrementEndpoint:
    def __init__(self, samples, N_rolling):
        samples['logZs'] = np.logaddexp.accumulate(samples.logw())
        self.logZs = samples.logZs
        self.samples = samples
        self.dlogZ = get_dlogZ(self.logZs)
        self.dlogZ_rolling = get_dlogZ_rolling(self.dlogZ, N_rolling)
        self.N_rolling = N_rolling
        self.rolling_index = self.dlogZ_rolling.index.get_level_values(0)
        self.true_endpoint = self.calc_endpoint() # using default value of 1e-3

    def plot_dlogZ(self):
        dlogZ = self.dlogZ
        plt.plot(dlogZ.index.get_level_values(0), dlogZ.values)

    def calc_endpoint(self, epsilon=1e-3):
        logZs = self.logZs
        logZ_tot = logZs.iloc[-1]
        logZ_f = np.log(1 - epsilon) + logZ_tot
        index_f = logZs[logZs > logZ_f].index.get_level_values(0)[0]
        return index_f

    def index(self, iteration):
        return self.dlogZ.iloc[iteration:].index.get_level_values(0).values
    
    def dlogZ_fit(self, iteration, steps, step):
        index = np.arange(iteration - steps*step, iteration+step, step)
        logZs = self.logZs.iloc[index]
        return logZs.diff(1).dropna()
    
    def pred(self, iteration, steps, step):
        index = np.arange(iteration - steps*step, len(self.samples))
        dlogZ_fit = self.dlogZ_fit(iteration, steps, step)
        coefs = fit_dlogZ(dlogZ_fit, 1)
        dlogZ_pred = poly.polyval(index, coefs)
        index_pred = index[dlogZ_pred > 0]
        dlogZ_pred = dlogZ_pred[dlogZ_pred > 0]
        return index_pred, dlogZ_pred

    def plot_pred(self, iteration, steps, step):
        index_pred, dlogZ_pred = self.pred(iteration, steps, step)
        dlogZ_fit = self.dlogZ_fit(iteration, steps, step)
        dlogZ_rolling = self.logZs.iloc[::step].diff(1).dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.plot(dlogZ_rolling.index.get_level_values(0), dlogZ_rolling)
        ax1.plot(dlogZ_fit.index.get_level_values(0), dlogZ_fit, color='deepskyblue')
        ax1.plot(index_pred, dlogZ_pred, color='orange', lw=1)
        ax1.axvline(x = iteration, lw=.5, ls='--', color='deepskyblue')
        ax1.set_ylim(0, dlogZ_rolling.iloc[0])
        ax1.set_ylabel('$d\log Z$')

        ax2.plot(dlogZ_rolling.index.get_level_values(0), dlogZ_rolling)
        ax2.plot(dlogZ_fit.index.get_level_values(0), dlogZ_fit, color='deepskyblue')
        ax2.plot(index_pred, dlogZ_pred, color='orange', lw=1)
        ax2.axvline(x = iteration, lw=.5, ls='--', color='deepskyblue')
        ax2.set_xlim(iteration - steps*step, len(self.samples))
        ax2.set_ylim(0, dlogZ_fit.values[0]*1.5)
        plt.tight_layout()

    def iterations(self, iteration, steps, step, epsilon=1e-3):
        logZ_dead = self.logZs.loc[iteration]
        index_pred, dlogZ_pred = self.pred(iteration, steps, step)
        logZ_live = dlogZ_pred.sum()
        logZ_tot = logZ_dead + logZ_live
        logZ_f = np.log(1 - epsilon) + logZ_tot
        index_f = index_pred[np.argmax([logZ_dead + dlogZ_pred.cumsum() > logZ_f])]
        return logZ_dead, logZ_tot, index_f
    
    def predictions(self, N, steps, step):
        true_end = self.true_endpoint
        iterations = np.linspace(steps*step + self.N_rolling, true_end, N, endpoint=False).astype(int) # start at steps, step
        predictions = np.zeros(N)
        for i, iteration in enumerate(iterations):
            # try:
                # predictions[i] = self.iterations(iteration, steps, step)[-1]
            # except:
                # print(f'Iteration {iteration} invalid')
            predictions[i] = self.iterations(iteration, steps, step)[-1]
        return iterations, predictions
    
    def plot_predictions(self, N, steps, step):
        true_end = self.true_endpoint
        iterations, predictions = self.predictions(N, steps, step)
        plt.plot(iterations, predictions)
        plt.plot(iterations, iterations, lw=1, ls='--', color='deepskyblue')
        plt.axhline(y=true_end, lw=1, ls='--')
        plt.ylim(0, true_end*1.5)
        return iterations, predictions
