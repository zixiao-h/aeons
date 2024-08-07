import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc, logsumexp, gammaincinv, loggamma

proj_dir = '/home/zixiao/Documents/III/project'
aeons_dir = '/home/zixiao/Documents/III/project/aeons'
data_dir = f'{aeons_dir}/data'
lcdm_chains = ['BAO', 'lensing', 'lensing_BAO', 'lensing_SH0ES', 'SH0ES', 'planck', 'planck_BAO', \
              'planck_lensing', 'planck_SH0ES', 'planck_lensing_BAO', 'planck_lensing_SH0ES']
test_chains = ['gauss_32', 'cauchy_83', 'BAO', 'planck']
gauss_chains = ['gauss_4', 'gauss_8', 'gauss_16', 'gauss_32', 'gauss_64']
cauchy_chains = ['cauchy_82', 'cauchy_83', 'cauchy_84', 'cauchy_85']
all_chains = gauss_chains + cauchy_chains + lcdm_chains
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

def pickle_dump(filename, data):
    """Function that pickles data into a file"""
    import pickle, os
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle_out = open(filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_in(filename):
    import pickle
    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)
    return data


def write_to_txt(filename, data):
    import os
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Clear file
    open(filename, 'w').close()
    with open(filename, 'a') as f:
        for item in data:
            if not np.shape(item):
                item = [item]
            np.savetxt(f, np.array(item), newline=',')
            f.write('\n')

def read_from_txt(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(np.fromstring(line.rstrip('\n'), sep=','))
    return data

def get_samples(chain, reduced=True):
    if reduced:
        path = f"{aeons_dir}/data/samples/{chain}_reduced.pkl"
    else:
        path = f"{aeons_dir}/data/samples/{chain}.pkl"
    samples = pickle_in(path)
    return chain, samples

def reduce_samples(samples):
    samples = samples.drop_labels()
    samples = samples.loc[:, samples.columns.intersection(['nlive', 'logL', 'logL_birth'])]
    return samples

def save_samples(name, samples):
    samples_reduced = reduce_samples(samples)
    pickle_dump(f'{aeons_dir}/data/samples/{name}.pkl', samples)
    pickle_dump(f'{aeons_dir}/data/samples/{name}_reduced.pkl', samples_reduced)
    
def data(points):
    logL = points.logL.to_numpy()
    nk = points.nlive.to_numpy()
    X_mean = X_mu(nk)
    logZdead = points.logZ()
    return logL, X_mean, nk, logZdead 


def data_split(logLlive, Xlive, splits=1, trunc=None):
    if trunc is not None:
        logLlive, Xlive = logLlive[:-trunc], Xlive[:-trunc]
    start = len(Xlive) - int(len(Xlive)/splits)
    X_split, logL_split = Xlive[start:], logLlive[start:]
    return logL_split, X_split

def formatt(theta, sigfigs=2):
    logLmax, d, sigma = theta
    if isinstance(sigfigs, int):
        return f"[{logLmax:.{sigfigs}e}, {d:.{sigfigs}f}, {sigma:.{sigfigs}e}]"
    else:
        s1, s2, s3 = sigfigs
        return f"[{logLmax:.{s1}e}, {d:.{s2}f}, {sigma:.{s3}e}]"

def reject_outliers(data, degree=2):
    data = np.array(data)
    dev = np.abs(data - np.median(data))
    median_dev = np.median(dev)
    return data[dev < degree * median_dev]

def points_at_iteration(samples, ndead):
    nlive = samples.loc[ndead].nlive
    logL_k = samples.loc[ndead].logL
    points = samples.loc[samples.logL_birth < logL_k]
    nk = np.concatenate([points.nlive[:ndead], np.arange(nlive, 0, -1)])
    points = points.assign(nlive=nk)
    points = points.reset_index(drop=True)
    return points

def logX_mu(nk):
    """Calculates the mean of logX at each iteration given the live point distribution for a NS run"""
    return -(1/nk).cumsum()

def X_mu(nk):
    """Mean of X for a live point distribution nk through a run"""
    return np.cumprod(nk/(nk+1))

def logt_sample(n):
    """Generate logt for given number of live points n"""
    p = np.random.rand(len(n))
    return 1/n * np.log(p)

def generate_Xs(nk, iterations=None):
    """Generates the Xs at each iteration in a run with live point distribution nk"""
    logt_samples = logt_sample(nk)
    logXs = np.cumsum(logt_samples)
    Xs = np.exp(logXs)
    if iterations is not None:
        return np.take(Xs, iterations)
    return Xs

def logZ_formula(logPmax, H, D, details=False):
    if details:
        print(f"logPr_max: {logPmax}, Hessian: {- 1/2 * np.log(abs(np.linalg.det(H)))}")
    return logPmax - 1/2 * np.log(abs(np.linalg.det(H))) + D/2 * np.log(2*np.pi)

def logXf_formula(theta, logZdead, Xi, epsilon=1e-3):
    logLmax, d, sigma = theta
    loglive = loggamma(d/2) + np.log(gammainc(d/2, Xi**(2/d)/(2*sigma**2)) )
    logdead = logZdead - logLmax - (d/2)*np.log(2) - d*np.log(sigma) + np.log(2/d)
    logend = logsumexp([loglive, logdead]) + np.log(epsilon)
    # if (gammainc(d/2, Xi**(2/d)/(2*sigma**2)) > 1 - epsilon):
    if logend > loggamma(d/2):
        return d/2 * np.log(2) + d*np.log(sigma) + loggamma(1 + d/2) + np.log(epsilon)
    xf_reg = gammaincinv(d/2, np.exp(logend - loggamma(d/2)))
    return d/2 * np.log(2*sigma**2 * xf_reg)


def iterations_rem(logXs, logXfs, nlive):
    dlogX = logXfs - logXs
    iterations_rem = dlogX * -nlive
    return iterations_rem

def calc_true_endpoint(logZ, epsilon=1e-3):
    logZ_tot = logZ.iloc[-1]
    logZ_f = np.log(1 - epsilon) + logZ_tot
    index_f = logZ[logZ > logZ_f].index.get_level_values(0)[0]
    return index_f

def calc_endpoints(iterations, logXs, logXfs, logXfs_std, nlive, nconst=None, logXconst=None):
    """nlive can be an integer or an array"""
    if nconst is None:
        endpoints = iterations + (logXfs - logXs) * -nlive
    else:
        if logXconst is None:
            raise ValueError("logXconst must be specified if nconst is specified")
        endpoints = nconst + (logXfs - logXconst) * -nlive
    endpoints_std = logXfs_std * -nlive
    return endpoints, endpoints_std

def make_iterations(true_endpoint, N, start=0.01, end=1):
    return np.linspace(start*true_endpoint, end*true_endpoint, N, endpoint=False).astype(int)

def get_logXs(samples, iterations):
    return samples.logX().iloc[iterations]

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
    
def get_logw(logL, logX, beta):
    logL = np.array(logL)
    logX = np.array(logX)
    logw = beta*logL + logX
    return logw

def format_exp(num, decimals=1):
    if num == 0:
        return 0, 0
    exp = 0
    while abs(num) < 1:
        num *= 10
        exp -= 1
    while abs(num) >= 10:
        num /= 10
        exp += 1
    if num == 0:
        return r'0'
    if exp == 0:
        return f'{np.round(num, decimals)}'
    if decimals == 0:
        return f"{int(np.round(num, 0))}\\times 10^{{{exp}}}"
    else:
        return f"{np.round(num, decimals)}\\times 10^{{{exp}}}"
    
def get_beta(points, ndead):
    logX = points.logX().iloc[ndead]
    if logX < -points.D_KL():
        return 1
    def func(beta):
        return logX + points.set_beta(beta).D_KL()
    from scipy import optimize
    res = optimize.root_scalar(func, bracket=[0, 1])
    return res.root

def get_dGs(get_beta, samples, iterations, Nset=10, **kwargs):
    """Get the dG of the samples at each iteration for a given function get_beta."""
    iterations = iterations.astype(int)
    d_Gs = np.zeros((len(iterations), Nset))
    for i, iteration in enumerate(iterations):
        points = points_at_iteration(samples, iteration)
        beta = get_beta(points, iteration, **kwargs)
        points = points.set_beta(beta)
        d_Gs[i] = points.d_G(Nset)
        print('\r', f'Iteration {iteration} of {iterations[-1]}', end='')
    return d_Gs
