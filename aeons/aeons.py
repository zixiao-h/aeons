import numpy as np
from scipy.special import logsumexp, loggamma, gammaincinv, gammainc
from anesthetic import NestedSamples


def logt_sample(n):
    """Generate logt for given number of live points n"""
    p = np.random.rand(len(n))
    return 1/n * np.log(p)


def generate_Xs(nk):
    """Generates the Xs at each iteration in a run with live point distribution nk"""
    logt_samples = logt_sample(nk)
    logXs = np.cumsum(logt_samples)
    Xs = np.exp(logXs)
    return Xs


def logXf_formula(theta, logZdead, Xi, epsilon=1e-3):
    logLmax, d, sigma = theta
    loglive = loggamma(d/2) + np.log(gammainc(d/2, Xi**(2/d)/(2*sigma**2)) )
    logdead = logZdead - logLmax - (d/2)*np.log(2) - d*np.log(sigma) + np.log(2/d)
    logend = logsumexp([loglive, logdead]) + np.log(epsilon)
    if logend > loggamma(d/2):
        return d/2 * np.log(2) + d*np.log(sigma) + loggamma(1 + d/2) + np.log(epsilon)
    xf_reg = gammaincinv(d/2, np.exp(logend - loggamma(d/2)))
    return d/2 * np.log(2*sigma**2 * xf_reg)


def sigma_squared_analytic(d, X_i, logL_i):
    """Returns the best-fit sigma squared as a function of d and the live points at a certain iteration i"""
    n = len(X_i)
    logsum = np.sum(logL_i)
    sum_X_4d = np.sum(X_i**(4/d))
    sum_X_2d = np.sum(X_i**(2/d))
    sum_log_X_2d = np.sum(X_i**(2/d) * logL_i)
    numerator = n * sum_X_4d - sum_X_2d**2
    denominator = 2 * logsum * sum_X_2d - 2*n*sum_log_X_2d
    return numerator/denominator


def logLmax_analytic(d, X_i, logL_i):
    """Returns the best-fit logLmax as a function of d and the live points at a certain iteration i"""
    n = len(X_i)
    logsum = np.sum(logL_i)
    sum_X_2d = np.sum(X_i**(2/d))
    return 1/n * logsum + 1/(2*n*sigma_squared_analytic(d, X_i, logL_i)) * sum_X_2d


def params_from_d(logLdata, Xdata, d):
    """Calculates the best-fit (logLmax, d, sigma) from d using analytic expressions"""
    sigma = np.sqrt(sigma_squared_analytic(d, Xdata, logLdata))
    logL_max = logLmax_analytic(d, Xdata, logLdata)
    return [logL_max, d, sigma]


def get_logbeta_post(points, ndead):
    betas = np.logspace(-5, 1, 1000)
    logX = points.logX()
    logL = points.logL
    logXs = logX.iloc[ndead]
    logLs = logL.iloc[ndead]
    logLbetasX = betas * logLs + logXs - points.logZ(beta=betas) + np.log(betas)
    logprob = logLbetasX - logsumexp(logLbetasX)
    mean = np.sum(np.exp(logprob)*np.log(betas))
    var = np.sum(np.exp(logprob)*(np.log(betas)-mean)**2)
    return mean, np.sqrt(var)


def get_d_G_post(points, ndead, Nset=25):
    logbeta_mean, logbeta_std = get_logbeta_post(points, ndead)
    betas_post = np.exp(np.random.normal(logbeta_mean, logbeta_std, Nset))
    d_G_post = points.d_G(beta=betas_post)
    return d_G_post.values


def logXfs(ndead, logL, logL_birth, Nset=25):
    points = NestedSamples(logL=logL, logL_birth=logL_birth)
    nk = points.nlive
    X_mean = np.cumprod(nk/(nk+1))
    logZ_dead = points[:ndead].logZ()
    d_G = get_d_G_post(points, ndead)
    mean, std = d_G.mean(), d_G.std() # Summarise d_G with its mean and std
    
    logXfs = np.zeros(Nset)
    for i in range(Nset):
        # Take a sample of X
        X = generate_Xs(nk)
        # Repeat until a valid logXf is found
        logXf_i = np.nan
        while np.isnan(logXf_i):
            # Sample from the posterior of d_G
            d = np.random.normal(mean, std)
            # Analytically compute the parameters of the regression to the logL-X curve
            theta = params_from_d(logL[ndead:], X[ndead:], d)
            # Compute the logXf for this d
            logXf_i = logXf_formula(theta, logZ_dead, X_mean[ndead])
        logXfs[i] = logXf_i
    
    return logXfs