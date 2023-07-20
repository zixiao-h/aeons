import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def logL_model(theta, X):
    """
    Input: (theta, X)
    Output: (logLmax - X**(2/d)/(2*sigma**2))
    """
    logLmax, d, sigma = theta
    return logLmax - X**(2/d)/(2*sigma**2)


def levenberg_marquardt(logLdata, Xdata, theta0, bounds=([-np.inf, 0, 0], np.inf)):
    """
    Input: logLdata, Xdata, theta0
    Output: (solution), solution of parameters to least squares fit of logLdata vs Xdata using
            Levenberg-Marquardt, implemented by scipy.optimize.least_squares
    """
    def logL_loss(theta):
        return logLdata - logL_model(theta, Xdata)
    solution = least_squares(logL_loss, theta0, bounds=bounds)
    return solution


def live_data(samples, iteration):
    """
    Input: samples, iteration
    Output: live_logL, live_X at that iteration
    """
    indices = samples.live_points(iteration).index
    live_logL = samples.logL.iloc[indices]
    live_logX = samples.logX()[indices]
    live_X = np.exp(live_logX)
    return live_logL, live_X


def local_live_lm(samples, iteration, theta0=None):
    """Returns least squares estimate of parameters at given iteration of a run
    Finds minimum nearest to initial parameters"""
    live_logL, live_X = live_data(samples, iteration)
    theta = theta0 if theta0 else [0.5, 1, 0.001]
    solution = levenberg_marquardt(live_logL, live_X, theta, bounds=([-np.inf, 0, 0], [np.inf, samples.shape[1], np.inf]))
    return solution


def generate_theta0(i, repeats, dmax):
    """Generates initial conditions based on some index i"""
    return [0.5-1000*i, 1+i, 0.001+i*0.1]


def log_uniform_guesses(i, repeats, dmax):
    """Produces array initial guesses which are log-uniform in d and sigma; free to set dmax eg.
    to the number of parameters in the chain"""
    def log_uniform_ratio(a, b, n):
        logratio = np.log(b/a)/(n-1)
        return np.exp(logratio)
    # d set to fall within a sensible range
    d_bounds = [1, dmax]
    d_ratio = log_uniform_ratio(*d_bounds, repeats)
    # Sigma set to something sensible
    sigma_bounds = [1e-5, 0.1]
    sigma_ratio = log_uniform_ratio(*sigma_bounds, repeats)
    return [0.5, d_bounds[0]*(d_ratio)**i, sigma_bounds[0]*sigma_ratio**i]


def global_live_lm(samples, iteration, guesses, repeats):
    """Estimates global minimum solution by running local_live_lm on a spread of initial conditions"""
    min_cost = np.inf
    min_solution = None
    print(iteration)
    for i in range(repeats):
        dmax = samples.shape[1]
        theta0 = guesses(i, repeats, dmax)
        try:
            solution = local_live_lm(samples, iteration, theta0)
            if solution.cost < min_cost:
                min_cost = solution.cost
                min_solution = solution
        except:
            print(f"{theta0} not appropriate")
        print(i, theta0, solution.x, solution.cost)
    return min_solution


def estimate_iterations(samples, method, iterations, args):
    """Simple loop running a method for a given list of iterations
    Pass args as tuple"""
    logLmax_estimates = []
    d_estimates = []
    sigma_estimates = []
    for i in iterations:
        logLmax, d, sigma = method(samples, i, *args).x
        logLmax_estimates.append(logLmax)
        d_estimates.append(d)
        sigma_estimates.append(sigma)
        print(f"Iteration {i} complete")
    return logLmax_estimates, d_estimates, sigma_estimates



def live_total_iterations(samples, iteration):
    _, d, sigma = local_live_lm(samples, iteration).x
    X_end = X_end_formula(d, sigma)
    logX_end = np.log(X_end)
    samples.live_points(iteration)
    # Approximate expected number of live points by average up to iteration
    avg_nlive = np.average(samples.nlive[:iteration+1])
    return -1 * avg_nlive * logX_end


def X_end_formula(d, sigma):
    """Returns end value of X given estimates of d, sigma"""
    from scipy.special import gamma
    return 0.001 * gamma(1+d/2) * 2**(d/2) * sigma**d
X_end_formula = np.vectorize(X_end_formula)