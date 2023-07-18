import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc, logsumexp, gammaincinv

proj_dir = '/home/zixiao/Documents/III/project'
aeons_dir = '/home/zixiao/Documents/III/project/aeons'

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


def write_to_txt(filename, data):
    try:
        open(filename+'.txt', 'w').close()
    except:
        print(f'creating {filename}.txt')
    with open(filename+'.txt', 'a') as f:
        for item in data:
            if not np.shape(item):
                item = [item]
            np.savetxt(f, np.array(item), newline=',')
            f.write('\n')

def read_from_txt(filename):
    data = []
    with open(filename+'.txt', 'r') as f:
        for line in f:
            data.append(np.fromstring(line.rstrip('\n'), sep=','))
    return data

def get_samples(which='lcdm', chain='BAO'):
    filename = f'{aeons_dir}/samples/{which}/{which}_{chain}.pickle'
    samples = pickle_in(filename)
    name = f'{which}_{chain}'
    return name, samples


def reject_outliers(data):
    data = np.array(data)
    dev = np.abs(data - np.median(data))
    median_dev = np.median(dev)
    return data[dev < 2 * median_dev]


def points_at_iteration(samples, ndead):
    nlive = samples.iloc[ndead].nlive
    logL_k = samples.iloc[ndead].logL
    points = samples[samples.logL_birth < logL_k]
    nk = np.concatenate([points.nlive[:ndead], np.arange(nlive+1, 1, -1)])
    points = points.assign(nlive=nk)
    return points

def logX_mu(nk):
    """Calculates the mean of logX at each iteration given the live point distribution for a NS run"""
    return -(1/nk).cumsum()

def X_mu(nk):
    """Mean of X for a live point distribution nk through a run"""
    return np.cumprod(nk/(nk+1))

def logt_sample(n):
    """Generate logt for given number of live points n"""
    p = np.random.rand()
    return 1/n * np.log(p)
logt_sample = np.vectorize(logt_sample)


def generate_Xs(nk, iterations=None):
    """Generates the Xs at each iteration in a run with live point distribution nk"""
    logt_samples = logt_sample(nk)
    logXs = np.cumsum(logt_samples)
    Xs = np.exp(logXs)
    if iterations is not None:
        return np.take(Xs, iterations)
    return Xs


def logXf_formula(theta, logZdead, Xi, epsilon=1e-3):
    logLmax, d, sigma = theta
    loglive = np.log( gamma(d/2) * gammainc(d/2, Xi**(2/d)/(2*sigma**2)) )
    logdead = logZdead - logLmax - (d/2)*np.log(2) - d*np.log(sigma) + np.log(2/d)
    logend = logsumexp([loglive, logdead]) + np.log(epsilon)
    xf_reg = gammaincinv(d/2, np.exp(logend)/gamma(d/2))
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

def calc_endpoints(iterations, logXs, logXfs, logXfs_std, nlive):
    """nlive can be an integer or an array"""
    endpoints = iterations + iterations_rem(logXs, logXfs, nlive)
    endpoints_higher = iterations + iterations_rem(logXs, logXfs - logXfs_std, nlive)
    endpoints_std = endpoints_higher - endpoints
    return endpoints, endpoints_std


def add_logZ(samples):    
    logw = samples.logw()
    logZ = np.zeros_like(logw)
    logZ[0] = logw.iloc[0]
    for i in range(1, len(samples)):
        logZ[i] = logsumexp([logZ[i-1], logw.iloc[i]])
    samples['logZs'] = logZ
    return samples


    