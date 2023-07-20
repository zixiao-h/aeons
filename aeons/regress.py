import numpy as np
from scipy.optimize import least_squares
from aeons.likelihoods import full
from aeons.utils import logZ_formula


def sigma_squared_analytic(d, X_i, logL_i):
    """Sigma squared as a function of d and the live points at a certain iteration i"""
    n = len(X_i)
    logsum = np.sum(logL_i)
    sum_X_4d = np.sum(X_i**(4/d))
    sum_X_2d = np.sum(X_i**(2/d))
    sum_log_X_2d = np.sum(X_i**(2/d) * logL_i)
    numerator = n * sum_X_4d - sum_X_2d**2
    denominator = 2 * logsum * sum_X_2d - 2*n*sum_log_X_2d
    return numerator/denominator


def logLmax_analytic(d, X_i, logL_i):
    """Returns logLmax as a function of d and the live points at a certain iteration i"""
    n = len(X_i)
    logsum = np.sum(logL_i)
    sum_X_2d = np.sum(X_i**(2/d))
    return 1/n * logsum + 1/(2*n*sigma_squared_analytic(d, X_i, logL_i)) * sum_X_2d


def params_from_d(logLdata, Xdata, d):
    """Calculates (logLmax, sigma) from d using analytic expressions"""
    sigma = np.sqrt(sigma_squared_analytic(d, Xdata, logLdata))
    logL_max = logLmax_analytic(d, Xdata, logLdata)
    return [logL_max, d, sigma]


def analytic_lm(logLdata, Xdata, d0, bounds=(0, np.inf)):
    """
    Input: logLdata, Xdata, d0
    Output: (solution), solution of parameters to least squares fit of logLdata vs Xdata using
            Levenberg-Marquardt, implemented by scipy.optimize.least_squares
    """
    def logL_loss(d):
        return logLdata - (logLmax_analytic(d, Xdata, logLdata) - \
            (Xdata**(2/d)) / (2 * sigma_squared_analytic(d, Xdata, logLdata)) )
    solution = least_squares(logL_loss, d0, bounds=bounds)
    return solution
 

def analytic_lm_params(logLdata, Xdata, d0, bounds=(0, np.inf)):
    """Returns [logLmax, d, sigma] using analytic LM method"""
    d, = analytic_lm(logLdata, Xdata, d0, bounds=bounds).x
    return params_from_d(logLdata, Xdata, d)


class GaussianRegress:
    def __init__(self, logL, X):
        self.logL = np.array(logL)
        self.X = np.array(X)
        self.N = len(logL)
        try:
            self.theta = analytic_lm_params(self.logL, self.X, d0=1)
        except:
            print('Bad data')
            self.theta = np.nan
    
    def L_sq(self, theta):
        loss = self.logL - full.func(self.X, theta)
        return np.sum(loss**2)
    
    def s(self, theta):
        return np.sqrt(self.L_sq(theta)/self.N)
    
    def logPr(self, theta):
        L_sq = self.L_sq(theta)
        s = np.sqrt(L_sq/self.N)
        return -1/2 * self.N * np.log(2*np.pi*s**2) - L_sq/(2*s**2)
    
    def logZ(self, theta_max=None, details=False):
        if theta_max == None:
            theta_max = self.theta
        logPr_max = self.logPr(theta_max)
        H = self.hess(theta_max)
        D = len(theta_max) + 1
        return logZ_formula(logPr_max, H, D, details)
    
    def hess(self, theta_max=None):
        import torch
        from torch.autograd.functional import hessian
        if theta_max == None:
            theta_max = self.theta
        s = self.s(theta_max)
        logL = torch.from_numpy(self.logL)
        mean = torch.from_numpy(self.X)
        theta_s_max = torch.tensor([*theta_max, s], requires_grad=True)
        def func(theta_s):
            *theta, s = theta_s
            loss = logL - full.func(mean, theta, torched=True)
            L_sq = torch.sum(loss**2)
            return -1/2 * self.N * torch.log(2*torch.pi*s**2) - L_sq/(2*s**2)
        H = hessian(func, theta_s_max)
        return np.array(H)
    
    def covtheta(self, theta_max=None):
        """Redefine to exclude rows/columns with covariance of s"""
        if theta_max == None:
            theta_max = self.theta
        Dtheta = len(theta_max)
        H = self.hess(theta_max)
        return np.linalg.inv(-H[:Dtheta, :Dtheta])