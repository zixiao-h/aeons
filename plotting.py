import numpy as np
import matplotlib.pyplot as plt
from lm_full import live_data, logL_model, global_live_lm, estimate_iterations, generate_theta0
from ipywidgets import interact
import ipywidgets as widgets

def plot_estimates(iterations, d_estimates, sigma_estimates, **kwargs):
    """Plots d, sigma against iterations"""
    figsize = kwargs.get('figsize', (3,1))
    fontsize = kwargs.get('fontsize', 6)
    lw = kwargs.get('lw', 1)
    dpi = kwargs.get('dpi', 400)

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(1,2, figsize=figsize, dpi=dpi)
    axs[0].plot(iterations, d_estimates, lw=lw)
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylim(0, np.max(d_estimates)*1.1)
    axs[0].set_title('d')

    axs[1].plot(iterations, sigma_estimates, lw=lw)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylim(0, np.max(sigma_estimates)*1.1)
    axs[1].set_title(r'$\sigma$')
    plt.tight_layout()


def plot_fit_raw(Xdata, logLdata, model_estimates):
    """Plot real and modelled logL against X, given the model_estimates of [logLmax, d, sigma]"""
    logLmax_estimate, d_estimate, sigma_estimate = model_estimates
    plt.plot(Xdata, logLdata, label="data")
    plt.plot(Xdata, logL_model([logLmax_estimate, d_estimate, sigma_estimate], Xdata), label="model")
    plt.xlabel("$X$")
    plt.ylabel("$\log L$")
    plt.title(f"k={len(Xdata)}, d={d_estimate:.2f}, $\sigma$={sigma_estimate:.1e}")
    plt.legend()


def plot_fit_at_iteration(samples, iteration, estimates, **kwargs):
    """Plots actual vs modelled logL for an array of estimated parameters at a given iteration"""
    figsize = kwargs.get('figsize', (3,3))
    fontsize = kwargs.get('fontsize', 6)
    lw = kwargs.get('lw', 1)
    dpi = kwargs.get('dpi', 200)

    logLmax_estimate, d_estimate, sigma_estimate = estimates
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    logLdata, Xdata = live_data(samples, iteration)
    ax.plot(Xdata, logL_model((logLmax_estimate, d_estimate, sigma_estimate), Xdata), label='model', lw=lw)
    ax.plot(Xdata, logLdata, label='real', lw=lw)
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$\log L$")
    ax.set_title(f"i={iteration}, d={d_estimate:.2f}, $\sigma$={sigma_estimate:.1e}")
    plt.tight_layout()


def plot_fits_at_iterations(samples, iterations, estimates, **kwargs):
    """Plots actual vs modelled logL for an array of estimated parameters"""
    figsize = kwargs.get('figsize', (3,3))
    fontsize = kwargs.get('fontsize', 6)
    lw = kwargs.get('lw', 1)
    logLmax_estimates, d_estimates, sigma_estimates = estimates
    columns = 2
    if len(iterations) % columns == 0:
        rows = int(len(iterations)/columns)
    else:
        rows = int(len(iterations)/columns) + 1
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(rows, columns, figsize=figsize, dpi=400)
    axs = axs.flatten()
    for i, iteration in enumerate(iterations):
        logLdata, Xdata = live_data(samples, iteration)
        axs[i].plot(Xdata, logL_model((logLmax_estimates[i], d_estimates[i], sigma_estimates[i]), Xdata), label='model', lw=lw)
        axs[i].plot(Xdata, logLdata, label='real', lw=lw)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(r"$X$")
        axs[i].set_ylabel(r"$\log L$")
        axs[i].set_title(f"i={iteration}, d={d_estimates[i]:.2f}, $\sigma$={sigma_estimates[i]:.1e}")
    axs[0].legend()
    plt.tight_layout()



def contours(grids, min_likelihood=-1000):
    """Plots contours for adjustable values of logLmax given a (N, N, N) array of likelihood evaluations at the specified parameters
    Input parameter grids = [logLmaxs, ds, sigmas, logPrs] where the logLmaxs etc are arrays of where the function was evalutated"""
    logLmaxs, ds, sigmas, logPrs = grids
    N = len(logLmaxs)
    @interact(logLmax_index=widgets.IntSlider(min=0, max=N-1, step=1))
    def plot_dsigma_contour(logLmax_index):
        dv, sigmav = np.meshgrid(ds, sigmas)
        contour_range = np.linspace(min_likelihood, logPrs[logLmax_index].max(), 20)
        plt.contourf(dv, sigmav, logPrs[logLmax_index].T, contour_range);
        (d_max_index, sigma_max_index), max_val = np.unravel_index(logPrs[logLmax_index].argmax(), logPrs[logLmax_index].shape), logPrs[logLmax_index].max()
        dbest, sigmabest = ds[d_max_index], sigmas[sigma_max_index]
        plt.scatter(dbest, sigmabest, marker="x")
        plt.title(f"logLmax={logLmaxs[logLmax_index]:.1f}, d={dbest:.2f}, $\sigma$={sigmabest:.3f}, logPr={max_val:.2f}")


def plot_stats(samples, iterations, estimates):
    """Plots estimates of parameters and fit for each iteration"""
    logLmax_estimates, d_estimates, sigma_estimates = estimates
    columns = 3
    if len(iterations) % columns == 0:
        rows = int(len(iterations)/columns)
    else:
        rows = int(len(iterations)/columns) + 1
    
    # Plot fits
    figsize = (20,20)
    fig, axs = plt.subplots(rows + 1, columns, figsize=figsize)
    axs = axs.flatten()
    for i in range(columns):
        axs[i].remove()
    for i, iteration in enumerate(iterations):
        logLdata, Xdata = live_data(samples, iteration)
        axs[i+3].plot(Xdata, logLdata, label='real')
        axs[i+3].plot(Xdata, logL_model((logLmax_estimates[i], d_estimates[i], sigma_estimates[i]), Xdata), label='model')
        axs[i+3].set_title(f"Iteration {iteration}, d={d_estimates[i]:.2f}, $\sigma$={sigma_estimates[i]:.1e}")

    # Plot parameter estimates; two columns
    ax_d = plt.subplot(rows + 1, 2, 1)
    ax_d.plot(iterations, d_estimates)
    ax_d.set_ylim(0, np.max(d_estimates)*1.1)
    ax_d.set_title('d')
    
    ax_sigma = plt.subplot(rows + 1, 2, 2)
    ax_sigma.plot(iterations, sigma_estimates)
    ax_sigma.set_ylim(0, np.max(sigma_estimates)*1.1)
    ax_sigma.set_title(r'$\sigma$')

    #plt.subplots_adjust(wspace=1, hspace=1)


def uniform_iterations_array(samples, n_points):
    """Returns an array of n points equally spaced along the length of the samples chain,
    not including the start and end points"""
    return np.linspace(0, len(samples), n_points+2, dtype=int)[1:-1]


def plot_statistics_sample(samples, method, n_points, args):
    """Plot estimation statistics for <samples> using n equally spaced iterations,
    set number of repeats for LM to run to find global minimum"""
    iterations = uniform_iterations_array(samples, n_points)
    estimates = estimate_iterations(samples, method, iterations, args)
    plot_stats(samples, iterations, estimates)