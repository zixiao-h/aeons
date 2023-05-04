import numpy as np
from scipy.optimize import least_squares
# from aeons.lm_full import live_data


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


def analytic_lm_params(logLdata, Xdata, d0):
    """Returns [logLmax, d, sigma] using analytic LM method"""
    d, = analytic_lm(logLdata, Xdata, d0).x
    return params_from_d(logLdata, Xdata, d)


def analytic_lm_params_at_iteration(samples, iteration, d0=None):
    """Returns estimated parameters [logLmax, d, sigma] at a given iteration using LM
    with analytic simplification"""
    live_logL, live_X = live_data(samples, iteration)
    if not d0:
        d0 = 1
    d, = analytic_lm(live_logL, live_X, d0).x
    estimated_params = params_from_d(samples, iteration, d)
    return estimated_params


def X_end_formula(d, sigma, epsilon):
    """Returns end value of X given estimates of d, sigma"""
    from scipy.special import gamma
    return epsilon * gamma(1+d/2) * 2**(d/2) * sigma**d
X_end_formula = np.vectorize(X_end_formula)