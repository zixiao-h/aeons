import numpy as np
from scipy.optimize import least_squares, minimize


def logPr_gaussian(logL, likelihood, mean, covinv, theta):
    Xstar = likelihood.inverse(logL, theta)
    return - 1/2 * (Xstar - mean).T @ covinv @ (Xstar - mean) 


def logPr_bayes(logL, likelihood, mean, covinv, theta):
    """likelihood = f(X_i, theta)"""
    Xstar = likelihood.inverse(logL, theta)
    log_abs_fprimes = np.log(abs(likelihood.prime(Xstar, theta)))
    return - np.sum(log_abs_fprimes) - 1/2 * (Xstar - mean).T @ covinv @ (Xstar - mean)


def logPr_laplace(theta, logpr_max, theta_max, Hessian):
    if not hasattr(theta, "__len__"):
        return float(logpr_max - 1/2 * (theta - theta_max).T * (- Hessian) * (theta - theta_max))
    return logpr_max - 1/2 * (theta - theta_max).T @ (- Hessian) @ (theta - theta_max) # A = negative hessian


def minimise_ls(logL, likelihood, mean, theta0, bounds=(-np.inf, np.inf)):
    def loss(theta):
        return mean - likelihood.inverse(logL, theta)
    solution = least_squares(loss, theta0, bounds=bounds)
    return solution


def minimise_gaussian(logL, likelihood, mean, covinv, x0):
    def func(theta):
        return - logPr_gaussian(logL, likelihood, mean, covinv, theta)
    solution = minimize(func, x0, method='Nelder-Mead')
    return solution


def minimise_bayes(logL, likelihood, mean, covinv, x0):
    def func(theta):
        return - logPr_bayes(logL, likelihood, mean, covinv, theta)
    solution = minimize(func, x0, method='Nelder-Mead')
    return solution