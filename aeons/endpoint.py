import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
        self.samples = add_logZ(samples)
        self.logZ = self.samples['logZ']
        self.logX_mean = np.array(samples.logX())
        self.logL = np.array(samples.logL)
        self.L = np.exp(self.logL)
        self.nlive = float(samples.nlive[0])

    def points(self, ndead):
        return points_at_iteration(self.samples, ndead)
    
    def data(self, ndead):
        points = points_at_iteration(self.samples, ndead)
        logL = np.array(points.logL)
        nk = np.array(points.nlive)
        logZdead = self.logZ.iloc[ndead]
        return logL, nk, logZdead
        
    def calc_endpoint(self, epsilon=1e-3):
        logZ = self.logZ
        logZ_tot = logZ.iloc[-1]
        logZ_f = np.log(1 - epsilon) + logZ_tot
        index_f = logZ[logZ > logZ_f].index.get_level_values(0)[0]
        return index_f
    
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
                print(f'Iteration {ndead}/{iterations[-1]}')
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
        logL_norm = self.logL - self.logL.max()
        L_norm = np.exp(logL_norm)
        ax1.plot(self.logX_mean, L_norm)
        ax2.plot(self.logX_mean, L_norm*np.exp(self.logX_mean))
        ax1.set_title(f'logL_max = {self.logL.max():.2f}')



def add_logZ(samples):    
    logw = samples.logw()
    logZ = np.zeros_like(logw)
    logZ[0] = logw.iloc[0]
    for i in range(1, len(samples)):
        logZ[i] = logsumexp([logZ[i-1], logw.iloc[i]])
    samples['logZ'] = logZ
    return samples

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

def plot_endpoints(iterations, endpoints, endpoints_std, true_endpoint, ylim=1.1, logX_vals=None):
    # plt.plot(iterations, endpoints, lw=1, color='navy')
    if logX_vals is not None:
        iterations = -logX_vals
        xlim = logX_vals[-1]
    plt.fill_between(iterations, endpoints - endpoints_std, endpoints + endpoints_std, alpha=1, color='deepskyblue')
    plt.fill_between(iterations, endpoints - 2*endpoints_std, endpoints + 2*endpoints_std, alpha=.2, color='deepskyblue')
    plt.axhline(y=true_endpoint, lw=1, color='navy')
    plt.ylim(0, true_endpoint*ylim)

class IncrementEndpoint:
    def __init__(self, samples, N_rolling):
        self.samples = add_logZ(samples)
        self.logZ = self.samples['logZ']
        self.dlogZ = get_dlogZ(self.logZ)
        self.dlogZ_rolling = get_dlogZ_rolling(self.dlogZ, N_rolling)
        self.N_rolling = N_rolling
        self.rolling_index = self.dlogZ_rolling.index.get_level_values(0)
        self.true_endpoint = self.calc_endpoint() # using default value of 1e-3

    def plot_dlogZ(self):
        dlogZ = self.dlogZ
        plt.plot(dlogZ.index.get_level_values(0), dlogZ.values)

    def calc_endpoint(self, epsilon=1e-3):
        logZ = self.logZ
        logZ_tot = logZ.iloc[-1]
        logZ_f = np.log(1 - epsilon) + logZ_tot
        index_f = logZ[logZ > logZ_f].index.get_level_values(0)[0]
        return index_f

    def index(self, iteration):
        return self.dlogZ.iloc[iteration:].index.get_level_values(0).values
    
    def dlogZ_fit(self, iteration, N_fit):
        return self.dlogZ_rolling.iloc[iteration - N_fit - self.N_rolling : iteration - self.N_rolling]
    
    def pred(self, iteration, N_fit):
        index = np.arange(iteration - N_fit, len(self.samples))
        dlogZ_fit = self.dlogZ_fit(iteration, N_fit)
        coefs = fit_dlogZ(dlogZ_fit, 1)
        dlogZ_pred = poly.polyval(index, coefs)
        index_pred = index[dlogZ_pred > 0]
        dlogZ_pred = dlogZ_pred[dlogZ_pred > 0]
        return index_pred, dlogZ_pred

    def plot_pred(self, iteration, N_fit):
        index_pred, dlogZ_pred = self.pred(iteration, N_fit)
        dlogZ_rolling = self.dlogZ_rolling
        dlogZ_fit = self.dlogZ_fit(iteration, N_fit)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.plot(dlogZ_rolling.index.get_level_values(0), dlogZ_rolling)
        ax1.plot(dlogZ_fit.index.get_level_values(0), dlogZ_fit, color='deepskyblue')
        ax1.plot(index_pred, dlogZ_pred, color='orange', lw=1)
        ax1.set_ylim(0, dlogZ_rolling.iloc[0])

        ax2.plot(dlogZ_rolling.index.get_level_values(0), dlogZ_rolling)
        ax2.plot(dlogZ_fit.index.get_level_values(0), dlogZ_fit, color='deepskyblue')
        ax2.plot(index_pred, dlogZ_pred, color='orange', lw=1)
        ax2.axvline(x = iteration, lw=.5, ls='--', color='deepskyblue')
        ax2.set_xlim(iteration - N_fit, len(self.samples))
        ax2.set_ylim(0, dlogZ_fit.values[0]*1.5)

    def iterations(self, iteration, N_fit, epsilon=1e-3):
        logZ_dead = self.logZ.loc[iteration]
        index_pred, dlogZ_pred = self.pred(iteration, N_fit)
        logZ_live = dlogZ_pred.sum()
        logZ_tot = logZ_dead + logZ_live
        logZ_f = np.log(1 - epsilon) + logZ_tot
        index_f = index_pred[np.argmax([logZ_dead + dlogZ_pred.cumsum() > logZ_f])]
        return logZ_dead, logZ_tot, index_f
    
    def predictions(self, N, N_fit):
        true_end = self.true_endpoint
        iterations = np.linspace(N_fit, true_end, N, endpoint=False).astype(int) # start at N_fit
        predictions = np.zeros(N)
        for i, iteration in enumerate(iterations):
            try:
                predictions[i] = self.iterations(iteration, N_fit)[-1]
            except:
                print(f'Iteration {iteration} invalid')
        return iterations, predictions
    
    def plot_predictions(self, N, N_fit):
        true_end = self.true_endpoint
        iterations, predictions = self.predictions(N, N_fit)
        plt.plot(iterations, predictions)
        plt.plot(iterations, iterations, lw=1, ls='--', color='deepskyblue')
        plt.axhline(y=true_end, lw=1, ls='--')
        plt.ylim(0, true_end*1.5)
        return iterations, predictions
    

def iterations_rem(logXs, logXfs, nlive):
    dlogX = logXfs - logXs
    iterations_rem = dlogX * -nlive
    return iterations_rem

def endpoints_calc(iterations, logXs, logXfs, logXfs_std, nlive):
    endpoints = iterations + iterations_rem(logXs, logXfs, nlive)
    endpoints_higher = iterations + iterations_rem(logXs, logXfs - logXfs_std, nlive)
    endpoints_std = endpoints_higher - endpoints
    return endpoints, endpoints_std