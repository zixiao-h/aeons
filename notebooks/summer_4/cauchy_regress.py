import numpy as np

def logcauchy(X, theta, torched=False):
    logLmax, c, gamma = theta
    if torched:
        import torch
        return logLmax - (1 + c)/2 * torch.log( 1 + X**(2/c)/(gamma**2) )
    return logLmax - (1 + c)/2 * np.log( 1 + X**(2/c)/gamma**2 )

def logLmax_cauchy(logLi, Xi, c, gamma):
    N = len(logLi)
    summand = logLi + (1 + c)/2 * np.log( 1 + Xi**(2/c)/gamma**2 )
    return np.sum(summand)/N

def cauchy_logXf(theta, logZdead, logX0):
    # Do likelihood prediction with theta
    logX_pred = np.linspace(logX0, -200, 2000)
    X_pred = np.exp(logX_pred)
    logL_pred = logcauchy(X_pred, theta)
    dlogX = abs(np.diff(logX_pred)[0])
    logsummands_pred = logL_pred + logX_pred + np.log(dlogX)
    logZ_deads_pred = np.logaddexp(logZdead, np.logaddexp.accumulate(logsummands_pred))
    logZ_tot_pred = logZ_deads_pred[-1]
    log_differences = logZ_deads_pred - logZ_tot_pred
    logZ_fracs_pred = np.log(1 - np.exp(log_differences), where=log_differences != 0)
    return logX_pred[np.argmin(np.abs(logZ_fracs_pred - np.log(1e-3)))]