import numba as nb
@nb.jit(nopython=True)
def brute_force_min(logLmaxs, ds, sigmas):
    """Returns the minimum of the function logPr over a given range of parameters"""
    logLmax_best = None
    d_best = None
    sigma_best = None
    logPr_max = -np.inf
    for i in range(len(logLmaxs)):
        logLmax = logLmaxs[i]
        for j in range(len(ds)):
            d = ds[j]
            for k in range(len(sigmas)):
                sigma = sigmas[k]
                logX = d/2 * np.log(2*sigma**2 *(logLmax - logL))
                logPr_i = - (logX - mean).T @ cov_inv @ (logX - mean)
                if logPr_i > logPr_max:
                    logPr_max = logPr_i
                    logLmax_best, d_best, sigma_best = logLmax, d, sigma
        print(logLmax)
    return [logLmax_best, d_best, sigma_best]