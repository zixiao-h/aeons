import numpy as np
from scipy.special import logsumexp
from aeons.utils import *
from aeons.regress import *

## Different betas

def get_logbeta_analytic(logXs, d, sigma):
    return np.log(d*sigma**2) - 2/d * logXs

def get_logbeta_grad(points, ndead, interval=100):
    logX = points.logX()
    logL = points.logL
    delta_logX = logX.iloc[ndead+interval] - logX.iloc[ndead-interval]
    delta_logL = logL.iloc[ndead+interval] - logL.iloc[ndead-interval]
    return np.log(-delta_logX/delta_logL)

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

def get_betas_DKL(samples):
    from scipy.interpolate import interp1d
    beta = np.logspace(-14, 14, 1000)
    logX = -samples.D_KL(beta=beta)
    beta = np.concatenate([[0], beta])
    logX = np.concatenate([[0], logX])
    f = interp1d(logX, beta)
    return f(samples.logX())

def get_betas_logL(samples):
    from scipy.interpolate import interp1d
    beta = np.logspace(-14, 14, 1000)
    logL = samples.logL_P(beta=beta)/beta
    i = np.argmin(logL < samples.logL.max())
    beta = np.concatenate([[0],beta[:i+1]])
    logL = np.concatenate([[samples.logL.min()],logL.iloc[:i+1]])
    f = interp1d(logL, beta)
    return f(samples.logL)

### d_G
def get_d_G_grad(points, ndead, Nset=25):
    logbeta = get_logbeta_grad(points, ndead)
    beta = np.exp(logbeta)
    d_G = points.d_G(Nset, beta=beta)
    return d_G.values, ""

def get_d_G_logL(points, ndead, Nset=25):
    beta = points.beta_logL[ndead]
    d_G = points.d_G(Nset, beta=beta)
    return d_G.values, ""

def get_d_G_post(points, ndead, Nset=25):
    logbeta_mean, logbeta_std = get_logbeta_post(points, ndead)
    betas_post = np.exp(np.random.normal(logbeta_mean, logbeta_std, Nset))
    d_G_post = points.d_G(beta=betas_post)
    return d_G_post.values, f"{len(betas_post)} samples"

def get_d_G_prop(points, ndead):
    betas = np.logspace(-10, 0, 1000)
    d_G = points.d_G(beta=betas)
    Pbeta = d_G
    mean = np.average(d_G, weights=Pbeta)
    std = np.sqrt(float(np.cov(d_G, aweights=Pbeta)))
    d_G_prop = np.random.normal(mean, std, size=100)
    d_G_prop = d_G_prop[d_G_prop > 0]
    return d_G_prop, f"{len(d_G_prop)} samples"

## Endpoints

def logXfs_prop(points, ndead, Nset=25):
    d_G = get_d_G_prop(points, ndead)[0]
    mean, std = d_G.mean(), d_G.std()
    
    logL, X_mean, nk, logZdead = data(points)
    logXf_set = np.zeros(Nset)
    for i in range(Nset):
        X = generate_Xs(nk)
        d = np.random.normal(mean, std)
        theta = params_from_d(logL[ndead:], X[ndead:], d)
        while np.isnan(logXf_formula(theta, logZdead, X_mean[ndead])):
            d = np.random.normal(mean, std)
            theta = params_from_d(logL[ndead:], X[ndead:], d)
        logXf_set[i] = logXf_formula(theta, logZdead, X_mean[ndead])    
    logXf_set = logXf_set[~np.isnan(logXf_set)]
    return logXf_set, ""

def logXfs_post(points, ndead, Nset=25):
    d_G = get_d_G_post(points, ndead)[0]
    mean, std = d_G.mean(), d_G.std()
    
    logL, X_mean, nk, logZdead = data(points)
    logXf_set = np.zeros(Nset)
    for i in range(Nset):
        X = generate_Xs(nk)
        d = np.random.normal(mean, std)
        theta = params_from_d(logL[ndead:], X[ndead:], d)
        while np.isnan(logXf_formula(theta, logZdead, X_mean[ndead])):
            d = np.random.normal(mean, std)
            theta = params_from_d(logL[ndead:], X[ndead:], d)
        logXf_set[i] = logXf_formula(theta, logZdead, X_mean[ndead])  
    logXf_set = logXf_set[~np.isnan(logXf_set)]
    # logXf_set = reject_outliers(logXf_set)
    return logXf_set, f"{len(logXf_set)} samples, {d_G.mean():.1f}"