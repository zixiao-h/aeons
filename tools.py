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
    returns [logLmax_estimates, d_estimates, sigma_estimates]"""
    params_estimates = []
    for iteration in iterations_array:
        params_i = estimation_method(samples, iteration)
        params_estimates.append(params_i)
        print(f"Iteration {iteration} complete")
    return np.array(params_estimates)


def plot_params(iterations, params_estimates, **kwargs):
    """Plots d, sigma against iterations given a list of estimated parameters"""
    d_estimates = params_estimates[:,1]
    sigma_estimates = params_estimates[:,2]
    figsize = kwargs.get('figsize', (3,1))
    fontsize = kwargs.get('fontsize', 6)
    lw = kwargs.get('lw', 1)

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(1,2, figsize=figsize, dpi=400)
    axs[0].plot(iterations, d_estimates, lw=lw)
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylim(0, np.max(d_estimates)*1.1)
    axs[0].set_title('d')

    axs[1].plot(iterations, sigma_estimates, lw=lw)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylim(0, np.max(sigma_estimates)*1.1)
    axs[1].set_title(r'$\sigma$')
    plt.tight_layout()