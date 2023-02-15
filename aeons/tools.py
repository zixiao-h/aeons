import numpy as np
import matplotlib.pyplot as plt

def pickle_dump(filename, data):
    """Function that pickles data into a file"""
    import pickle
    pickle_out = open(filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_in(filename):
    import pickle
    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)
    return data


def params_at_iterations(samples, iterations_array, estimation_method):
    """Simple loop running an estimation method for a given list of iterations,
    returns [logLmax_estimates, d_estimates, sigma_estimates]
    Estimation method must take arguments (samples, iteration)"""
    params_estimates = []
    for iteration in iterations_array:
        params_i = estimation_method(samples, iteration)
        params_estimates.append(params_i)
        print(f"Iteration {iteration} complete")
    return np.array(params_estimates)
    