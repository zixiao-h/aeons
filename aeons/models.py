import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch.autograd.functional import hessian
from aeons.bayes import logPr_laplace
from aeons.true_distribution import generate_Xs


def logZ(logPmax, H, D, details=False):
    if details:
        print(f"logPr_max: {logPmax}, Hessian: {- 1/2 * np.log(abs(np.linalg.det(H)))}")
    return logPmax - 1/2 * np.log(abs(np.linalg.det(H))) + D/2 * np.log(2*np.pi)


def freq_std(Model, nk, likelihood, mean, covinv, theta_true, repeats=20, give_thetas=False):
    thetas = []
    for i in range(20):
        X = generate_Xs(nk)
        y = likelihood.func(X, theta_true)
        model = Model(y, likelihood, mean, covinv)
        thetas.append(model.minimise(theta_true))
    if give_thetas:
        return thetas, np.std(thetas)
    return np.std(thetas)


class Model:
    def __init__(self, y, likelihood, mean, covinv=None):
        self.y = y
        self.likelihood = likelihood
        self.mean = mean
        self.covinv = covinv
        self.N = len(y)

    def minimise(self, x0, bounds=None, full_solution=False):
        def func(theta):
            return - self.logPr(theta)
        solution = minimize(func, x0, method='Nelder-Mead', bounds=bounds)
        if full_solution:
            return solution
        else:
            return solution.x
    
    def logZ(self, theta_max, details=False):
        logPr_max = self.logPr(theta_max)
        H = self.hess(theta_max)
        D = len(theta_max) + 1
        return logZ(logPr_max, H, D, details)
    
    def covtheta(self, theta_max):
        H = self.hess(theta_max)
        return np.linalg.inv(-H)
    
    def plot_hess(self, theta_max, index, lim=1e-2, points=1000, plot=True):
        logPr_max = self.logPr(theta_max)
        param_max = theta_max[index]
        params = np.linspace(param_max*(1-lim), param_max*(1+lim), points)
        logprs = np.zeros_like(params)
        logprs_laplace = np.zeros_like(params)
        H = self.hess(theta_max)
        for i, param in enumerate(params):
            theta_arr = np.copy(theta_max)
            theta_arr[index] = param
            logprs[i] = self.logPr(theta_arr)
            logprs_laplace[i] = logPr_laplace(param, logPr_max, param_max, H[index][index])
        if plot:
            plt.plot(params, logprs, color='black', label='true')
            plt.plot(params, logprs_laplace, color='orange', label='laplace')
            plt.legend();
        return logprs, logprs_laplace, params

    def plot_errors(self, X, theta_max, lines=100):
        y = self.y
        like = self.likelihood
        covtheta = self.covtheta(theta_max)
        if len(theta_max) == 1:
            sample_func = np.random.normal
            cov = float(np.sqrt(covtheta))
            theta_max = float(theta_max)
        else:
            sample_func = np.random.multivariate_normal
            cov = covtheta
        for i in range(lines):
            theta = sample_func(theta_max, cov)
            Xcn = like.inverse(y, theta)
            plt.plot(Xcn, y, lw=.5, color='gray', alpha=.2)
        plt.plot(X, y, 'x', ms=1, color='blue', label='true')
        plt.plot(like.inverse(y, theta_max), y, lw=1, color='black', label=f'maximum: {np.round(theta_max, 2)}')
        plt.legend();
        
    

class LS(Model):
    def __init__(self, y, likelihood, mean, covinv=None):
        super().__init__(y, likelihood, mean, covinv)
    
    def L_sq(self, theta):
        # loss = self.mean - self.likelihood.inverse(self.y, theta)
        loss = self.y - self.likelihood.func(self.mean, theta)
        return np.sum(loss**2)
    
    def s(self, theta):
        return np.sqrt(self.L_sq(theta)/self.N)

    def logPr(self, theta):
        L_sq = self.L_sq(theta)
        s = np.sqrt(L_sq/self.N)
        return -1/2 * self.N * np.log(2*np.pi*s**2) - L_sq/(2*s**2)
    
    def hess(self, theta_max):
        s = self.s(theta_max)
        y = torch.from_numpy(self.y)
        mean = torch.from_numpy(self.mean)
        theta_s_max = torch.tensor([*theta_max, s], requires_grad=True)
        def func(theta_s):
            if len(theta_s) == 2:
                theta, s = theta_s
            else:
                *theta, s = theta_s
            # loss = mean - self.likelihood.inverse(y, theta, torched=True)
            loss = y - self.likelihood.func(mean, theta, torched=True)
            L_sq = torch.sum(loss**2)
            return -1/2 * self.N * torch.log(2*torch.pi*s**2) - L_sq/(2*s**2)
        H = hessian(func, theta_s_max)
        return np.array(H)
    
    def covtheta(self, theta_max):
        """Redefine to exclude rows/columns with covariance of s"""
        Dtheta = len(theta_max)
        H = self.hess(theta_max)
        return np.linalg.inv(-H[:Dtheta, :Dtheta])
    

class CG(Model):
    def __init__(self, y, likelihood, mean, covinv):
        super().__init__(y, likelihood, mean, covinv)
        self.logdet_inv = np.linalg.slogdet(covinv)[1]
    
    def logPr(self, theta):
        Xstar = self.likelihood.inverse(self.y, theta)
        log_abs_fprimes = np.log(abs(self.likelihood.prime(Xstar, theta)))
        prefactor = 1/2 * self.logdet_inv - 1/2 * self.N * np.log(2*np.pi)
        return prefactor - np.sum(log_abs_fprimes) - 1/2 * (Xstar - self.mean).T @ self.covinv @ (Xstar - self.mean)
    
    def hess(self, theta_max):
        y = torch.from_numpy(self.y)
        mean = torch.from_numpy(self.mean)
        covinv = torch.from_numpy(self.covinv)
        theta_max = torch.tensor(theta_max, requires_grad=True)
        def func(theta):
            Xstar = self.likelihood.inverse(y, theta, torched=True)
            log_abs_fprimes = torch.log(abs(self.likelihood.prime(Xstar, theta, torched=True)))
            return - torch.sum(log_abs_fprimes) - 1/2 * (Xstar - mean).T @ covinv @ (Xstar - mean)
        H = hessian(func, theta_max)
        return np.array(H)
     

class CG_naive(Model):
    def __init__(self, y, likelihood, mean, covinv):
        super().__init__(y, likelihood, mean, covinv)
        self.logdet_inv = np.linalg.slogdet(covinv)[1]
    
    def logPr(self, theta):
        Xstar = self.likelihood.inverse(self.y, theta)
        return 1/2 * self.logdet_inv - 1/2 * (Xstar - self.mean).T @ self.covinv @ (Xstar - self.mean)
    
    def hess(self, theta_max):
        y = torch.from_numpy(self.y)
        mean = torch.from_numpy(self.mean)
        covinv = torch.from_numpy(self.covinv)
        theta_max = torch.tensor(theta_max, requires_grad=True)
        def func(theta):
            Xstar = self.likelihood.inverse(y, theta, torched=True)
            return - 1/2 * (Xstar - mean).T @ covinv @ (Xstar - mean)
        H = hessian(func, theta_max)
        return np.array(H)