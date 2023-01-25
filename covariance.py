import numpy as np


def logX_mean(samples, iteration):
    """Function that returns the mean vector for a given iteration of samples"""
    indices = samples.live_points(iteration).index
    logX_mean = samples.logX()[indices]
    return np.array(logX_mean)

def logX_cov(samples, iteration):
    """Function that returns the covariance matrix for a given iteration of samples"""
    nk = int(samples.iloc[iteration].nlive)
    nlive = int(samples.iloc[0].nlive)
    iterations = samples.live_points(iteration).index
    vars = np.zeros(nk)
    for i, iteration in enumerate(iterations):
        vars[i] = (iteration+1)/(nlive**2)
    cov = np.zeros((nk, nk))
    for i in range(nk):
        for j in range(i, nk):
            cov[i][j] = cov[j][i] = vars[i]
    return cov

def logX_model(logL, logLmax, d, sigma):
     """Returns logX as a function of logL, logLmax, d, sigma"""
     logLmax_array = logLmax * np.ones(len(logL))
     return d/2 * np.log(2*sigma**2 *(logLmax_array - logL))

def logPr(logX, mean, cov_inv):
    """Log of probability density for a set of logX samples, given its mean and covariance"""
    return - (logX - mean).T @ cov_inv @ (logX - mean)