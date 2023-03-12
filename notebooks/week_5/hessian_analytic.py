import numpy as np
import sympy as sp
import numba as nb
import torch
from aeons.likelihoods import full_like
full = full_like()


logLmax, d, sigma = sp.symbols('\log{L_\mathrm{max}} d \sigma')
X_i, logL_i, logL_j = sp.symbols('X_i \log{L_i} \log{L_j}')
mu_i, mu_j = sp.symbols('\mu_i, \mu_j')


def f(X_i, logLmax, d, sigma):
    return logLmax - X_i**(2/d)/(2*sigma**2)

def fprime(X_i, logLmax, d, sigma):
    return sp.diff(f(X_i, logLmax, d, sigma), X_i)

def abs_fprime(X_i, logLmax, d, sigma):
    return - fprime(X_i, logLmax, d, sigma)

def X_logLi(logL_i, logLmax, d, sigma):
    return (2*sigma**2 * (logLmax - logL_i))**(d/2)

def log_abs_fprime_Xstar(logL_i, logLmax, d, sigma):
    Xstar_i = X_logLi(logL_i, logLmax, d, sigma)
    return sp.log(abs_fprime(X_i, logLmax, d, sigma).subs(X_i, Xstar_i))

def cross_terms(logL_i, logL_j, logLmax, d, sigma):
    Xstar_i = X_logLi(logL_i, logLmax, d, sigma)
    Xstar_j = X_logLi(logL_j, logLmax, d, sigma)
    return (Xstar_i - mu_i) * (Xstar_j - mu_j)

def hess_fprime_i(theta_1, theta_2):
    symbolic_expr = sp.simplify(sp.diff(log_abs_fprime_Xstar(logL_i, logLmax, d, sigma), theta_1, theta_2))
    numeric_func = sp.lambdify([logL_i, logLmax, d, sigma], symbolic_expr)
    return numeric_func

def hess_cross_ij(theta_1, theta_2):
    symbolic_expr = sp.simplify(sp.diff(cross_terms(logL_i, logL_j, logLmax, d, sigma), theta_1, theta_2))
    numeric_func = sp.lambdify([logL_i, logL_j, mu_i, mu_j, logLmax, d, sigma], symbolic_expr)
    return numeric_func

def hess_logfprime(logL, theta_max, theta_1, theta_2):
    if set([theta_1, theta_2]) == set([logLmax, sigma]):
        return 0
    hess_fprime_i_array = hess_fprime_i(theta_1, theta_2)(logL, *theta_max) * np.ones_like(logL) # ensures still array-like if independent of logL_i
    return np.sum(hess_fprime_i_array)

def hess_cross(logL, mean, covinv, theta_max, theta_1, theta_2):
    hess_tt_ij = hess_cross_ij(theta_1, theta_2)
    k = len(logL)
    @nb.jit
    def quad():
        quad = 0
        for i in range(k):
            cross_i = hess_tt_ij(logL[i], logL, mean[i], mean, *theta_max) * covinv[i]
            quad += cross_i.sum()
        return quad
    return quad()

def hess_tt(logL, mean, covinv, theta_max, theta_1, theta_2):
    return - hess_logfprime(logL, theta_max, theta_1, theta_2) - 1/2 * hess_cross(logL, mean, covinv, theta_max, theta_1, theta_2)


def hess(logL, mean, covinv, theta_max):
    dim = 3
    hess = np.zeros((dim, dim))
    hess[0][0] = hess_tt(logL, mean, covinv, theta_max, logLmax, logLmax)
    hess[0][1] = hess[1][0] = hess_tt(logL, mean, covinv, theta_max, logLmax, d)
    hess[0][2] = hess[2][0] = hess_tt(logL, mean, covinv, theta_max, logLmax, sigma)
    hess[1][1] = hess_tt(logL, mean, covinv, theta_max, d, d)
    hess[1][2] = hess[2][1] = hess_tt(logL, mean, covinv, theta_max, sigma, d)
    hess[2][2] = hess_tt(logL, mean, covinv, theta_max, sigma, sigma)
    return hess


def hess_autograd(logL, likelihood, mean, covinv, theta_max):
    logL = torch.from_numpy(logL)
    mean = torch.from_numpy(mean)
    covinv = torch.from_numpy(covinv)
    input = torch.tensor(theta_max, requires_grad=True)
    
    def logPr_bayes(logL, likelihood, mean, covinv, theta):
        """likelihood = f(X_i, theta)"""
        Xstar = likelihood.inverse(logL, theta)
        log_abs_fprimes = torch.log(abs(likelihood.prime(Xstar, theta)))
        return - torch.sum(log_abs_fprimes) - 1/2 * (Xstar - mean).T @ covinv @ (Xstar - mean)

    def logPr_bayes_max(theta):
        return logPr_bayes(logL, likelihood, mean, covinv, theta)
    
    from torch.autograd.functional import hessian
    H = hessian(logPr_bayes_max, input)
    return np.array(H)