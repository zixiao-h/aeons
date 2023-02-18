import numpy as np
import numba as nb

def data_at_iteration(samples, iteration):
    points = points_at_iteration(samples, iteration)
    nk = np.array(points.nlive)
    logL = np.array(points.logL)
    return nk, logL 


def logX_model(logL, logLmax, d, sigma):
     """Returns logX as a function of logL, logLmax, d, sigma"""
     logLmax_array = logLmax * np.ones(len(logL))
     return d/2 * np.log(2*sigma**2 *(logLmax_array - logL))


def logPr(logX, mean, cov_inv):
    """Log of probability density for a set of logX samples, given its mean and covariance"""
    return - (logX - mean).T @ cov_inv @ (logX - mean)


def logPr_params(logL, mean, cov_inv, params):
    logLmax, d, sigma = params
    logX = logX_model(logL, logLmax, d, sigma)
    return logPr(logX, mean, cov_inv)


def X_logL(points):
    """Gets the X and logL arrays for a set of samples"""
    logL = points.logL
    X = np.exp(points.logX())
    return X, logL


def points_at_iteration(samples, iteration):
    logL_k = samples.iloc[iteration].logL
    all_points = samples[samples.logL_birth < logL_k]
    points = all_points.recompute()
    return points


@nb.jit(nopython=True)
def logX_mu(live_point_distribution):
    """Calculates the mean of logX at each iteration given the live point distribution for a NS run"""
    logX_mean = - (1/live_point_distribution).cumsum()
    return logX_mean


@nb.jit(nopython=True)
def logX_covinv_chol(live_point_distribution):
    """Calculates the covariance matrix between the logXs at each iteration given the live point distribution for a NS run
    using Cholesky decomposition"""
    total_iterations = len(live_point_distribution)
    L_inv = np.zeros((total_iterations, total_iterations))

    # Fill out L_ij using L_ij_inv = n_j for i=j, -n_j+1 for i = j+1. Skip last iteration and fill in at end to avoid out of range index
    for j in range(total_iterations - 1):
        L_inv[j][j] = live_point_distribution[j]
        L_inv[j+1][j] = - live_point_distribution[j+1]
    L_inv[total_iterations - 1][total_iterations - 1] = live_point_distribution[total_iterations - 1]
    
    covinv = L_inv.T @ L_inv
    return covinv


@nb.jit(nopython=True)
def logX_covinv_rud(live_point_distribution):
    """Calculates the covariance matrix between the logXs at each iteration given the live point distribution for a NS run
    by direct construction of covariance then finding inverse"""
    total_iterations = len(live_point_distribution)
    vars = (1/live_point_distribution**2).cumsum()
    cov = np.zeros((total_iterations, total_iterations))
    for i in range(total_iterations):
        for j in range(i, total_iterations):
            cov[i][j] = cov[j][i] = vars[i]
    cov_inv = np.linalg.inv(cov)
    return cov_inv


@nb.jit(nopython=True)
def X_mu(nk):
    """Mean of X for a live point distribution nk through a run"""
    return np.cumprod(nk/(nk+1))


@nb.jit(nopython=True)
def X_Sigma(nk):
    """Covariance matrix between X for a live point distribution nk through a run"""
    t_1 = np.cumprod(nk/(nk+1)) # cumulative product of expectation of t
    t_2 = np.cumprod(nk/(nk+2)) # cumulative product of expectation of t^2
    iterations = len(nk)
    cov_X = np.zeros((iterations, iterations))
    for i in range(iterations):
        cov_X[i][i] = t_2[i] - t_1[i]**2 
        for j in range(i+1, iterations): # start j above i so min(i,j) automatically fulfilled
            correlated = t_2[i] - t_1[i]**2
            independent = t_1[j]/t_1[i] # cumulative product from i+1 to j
            cov_X[i][j] = cov_X[j][i] = correlated * independent
    return cov_X


def X_Sigmainv(nk):
    """Inverse covariance between X for live point distribution nk through a nested sampling run"""
    return np.linalg.inv(X_Sigma(nk))


from scipy.optimize import minimize
def optimise_pr_cg(logL, mean, cov_inv, x0):
    """Optimise correlated gaussian probability as a function of the parameters (logLmax, d, sigma)
    
    Inputs: logL (datapoints), mean, cov_inv of the areas logX, initial guess for the parameters x0
    Outputs: solution object for parameters (logLmax, d, sigma)"""
    def func(theta):
        logX = logX_model(logL, *theta)
        return - logPr(logX, mean, cov_inv) # want to maximise probability <-> minimise negative
    solution = minimize(func, x0)
    return solution


def grid_search(logL, mean, covinv, params_range):
    """Brute force search of parameter range. Returns array of logPrs as 3D array"""
    logLmaxs, ds, sigmas = params_range
    NL, Nd, Ns = len(logLmaxs), len(ds), len(sigmas)

    @nb.jit(nopython=True)
    def brute():
        """Calculates all function values across some parameter space (can be specified by kwargs) and returns as an array"""
        def logX_model(logL, logLmax, d, sigma):
            logLmax_array = logLmax * np.ones(len(logL))
            return d/2 * np.log(2*sigma**2 *(logLmax_array - logL))

        def logPr(logX, mean, cov_inv):
            return - (logX - mean).T @ cov_inv @ (logX - mean)

        def logPr_params(logL, mean, cov_inv, params):
            logLmax, d, sigma = params
            logX = logX_model(logL, logLmax, d, sigma)
            return logPr(logX, mean, cov_inv)

        logPrs = np.zeros((NL, Nd, Ns))
        for i in range(NL):
            for j in range(Nd):
                for k in range(Ns):
                    params = [logLmaxs[i], ds[j], sigmas[k]]
                    logPrs[i][j][k] = logPr_params(logL, mean, covinv, params)
        return logPrs
    return brute()


def get_max(grids):
    """Get maximum likelihood and corresponding parameters from brute force evaluation of parameter space"""
    logLmaxs, ds, sigmas, logPrs = grids
    logLmax_index, d_index, sigma_index = np.unravel_index(np.argmax(logPrs), logPrs.shape)
    logLmax_best, d_best, sigma_best = logLmaxs[logLmax_index], ds[d_index], sigmas[sigma_index]
    logPr_max = logPrs.max()
    return logLmax_best, d_best, sigma_best, logPr_max

# def logX_mean_at_iteration(samples, iteration):
#     """Function that returns the mean vector for a given iteration of samples"""
#     indices = samples.live_points(iteration).index
#     logX_mean = samples.logX()[indices]
#     return np.array(logX_mean)


# @nb.jit(nopython=True)
# def cov_inv(nk, nlive, iterations):
#     """Returns covariance matrix between areas given the iteration numbers of the live points, default nlive and nk"""
#     vars = np.zeros(nk)
#     for i, iteration in enumerate(iterations):
#         vars[i] = (iteration+1)/(nlive**2)
#     cov = np.zeros((nk, nk))
#     for i in range(nk):
#         for j in range(i, nk):
#             cov[i][j] = cov[j][i] = vars[i]
#     cov_inv = np.linalg.inv(cov)
#     return cov_inv


# def cov_inv_at_iteration(samples, iteration):
#     """Function that returns the covariance matrix for a given iteration of samples"""
#     nk = int(samples.iloc[iteration].nlive)
#     nlive = int(samples.iloc[0].nlive)
#     iterations = np.array(samples.live_points(iteration).index)
#     return cov_inv(nk, nlive, iterations)


# def params_at_iteration_cg(samples, iteration):
#     """Returns estimated parameters for a given iteration of the samples, using the correlated gaussian
#     model of logX"""
#     live_points = samples.live_points(iteration)
#     logL = live_points.logL
#     mean = logX_mean(samples, iteration)
#     cov_inv = cov_inv_at_iteration(samples, iteration)

#     # Use previous estimate of x0 as starting point
#     from lm_partial import analytic_lm_params
#     x0 = analytic_lm_params(samples, iteration)
#     estimated_params = optimise_pr_cg(logL, mean, cov_inv, x0).x
#     return estimated_params