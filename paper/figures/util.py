import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf

endpoints = {'BAO': 18129,
 'lensing': 19034,
 'lensing_BAO': 22108,
 'lensing_SH0ES': 20672,
 'SH0ES': 15096,
 'planck': 57372,
 'planck_BAO': 58216,
 'planck_lensing': 58100,
 'planck_SH0ES': 57424,
 'planck_lensing_BAO': 57852,
 'planck_lensing_SH0ES': 58032,
 'gauss_4': 11672,
 'gauss_8': 18665,
 'gauss_16': 31329,
 'gauss_32': 53346,
 'gauss_64': 90737,
 'cauchy_82': 20961,
 'cauchy_83': 29889,
 'cauchy_84': 39236,
 'cauchy_85': 48405,
 'slab_spike': 6593,
 'correlated_3d': 26107,
 'correlated_6d': 30234,
 'gp': 91721,
 'gauss_32_2000': 213249,
 'gauss_16_2000': 125308,
 'gauss_16_1000': 62973}
DKLs = {'cauchy_85': 76.17846347858264,
 'cauchy_82': 21.72038931529883,
 'cauchy_83': 39.424475851934695,
 'cauchy_84': 57.11849802882677}

def read_from_txt(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(np.fromstring(line.rstrip('\n'), sep=','))
    return data

def figsettings():
    format = {
        # Set figure size to (4,2)
        "figure.figsize": (4, 2),
        # Use LaTeX to write all text
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        'font.family': 'serif',
        # Use 11pt font in plots, to match 11pt font in document
        "axes.labelsize": 7,
        "font.size": 7,
        "axes.titlesize": 9,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        'axes.linewidth': 0.5,
        'patch.linewidth': 0.5,
        'legend.fancybox': False,
        'legend.shadow': False
    }
    plt.rcParams.update(format)
    
def plot_std(xvals, y_means, y_stds, true=None, ylim=None, ax=None, color='deepskyblue', label=None):
    # plt.plot(xvals, y_means, lw=1, color='navy')
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,2))
    ax.fill_between(xvals, y_means - y_stds, y_means + y_stds, alpha=1, color=color, label=label)
    ax.fill_between(xvals, y_means - 2*y_stds, y_means + 2*y_stds, alpha=.2, color=color)
    if true is not None:
        ax.axhline(y=true, lw=.5, color='red', ls='--')
        if isinstance(ylim, float):
            ax.set_ylim(0, true*ylim)
        elif isinstance(ylim, tuple):
            ax.set_ylim(ylim[0]*true, ylim[1]*true)
            
def quantile(a, q, w=None, interpolation='linear'):
    """Compute the weighted quantile for a one dimensional array."""
    if w is None:
        w = np.ones_like(a)
    a = np.array(list(a))  # Necessary to convert pandas arrays
    w = np.array(list(w))  # Necessary to convert pandas arrays
    i = np.argsort(a)
    c = np.cumsum(w[i[1:]]+w[i[:-1]])
    c = c / c[-1]
    c = np.concatenate(([0.], c))
    icdf = interp1d(c, a[i], kind=interpolation)
    quant = icdf(q)
    if isinstance(q, float):
        quant = float(quant)
    return quant

def quantile_plot_interval(q):
    """Interpret quantile ``q`` input to quantile plot range tuple."""
    if isinstance(q, str):
        sigmas = {'1sigma': 0.682689492137086,
                  '2sigma': 0.954499736103642,
                  '3sigma': 0.997300203936740,
                  '4sigma': 0.999936657516334,
                  '5sigma': 0.999999426696856}
        q = (1 - sigmas[q]) / 2
    elif isinstance(q, int) and q >= 1:
        q = (1 - erf(q / np.sqrt(2))) / 2
    if isinstance(q, float) or isinstance(q, int):
        if q > 0.5:
            q = 1 - q
        q = (q, 1-q)
    return tuple(np.sort(q))

def plot_quantiles(x, y, true=None, ylim=None, label=None, ax=None, color='deepskyblue'):
    q1 = quantile_plot_interval(1)
    q2 = quantile_plot_interval(2)
    y1 = np.apply_along_axis(quantile, 1, y, q1).T
    y2 = np.apply_along_axis(quantile, 1, y, q2).T
    if ax is None:
        fig, ax = plt.subplots()
    ax.fill_between(x, *y1, alpha=0.8, color=color, label=label)
    ax.fill_between(x, *y2, alpha=0.2, color=color)
    if true is not None:
        ax.axhline(y=true, lw=.5, color='red', ls='--')
        if isinstance(ylim, float):
            ax.set_ylim(0, true*ylim)
        elif isinstance(ylim, tuple):
            ax.set_ylim(ylim[0]*true, ylim[1]*true)