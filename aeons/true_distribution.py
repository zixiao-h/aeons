import numpy as np

def nk_dead(nlive, ndead):
    return nlive * np.ones(ndead)


def nk_dead_live(nlive, ndead):
    nk0 = nlive * np.ones(ndead)
    nk1 = np.flip(np.arange(1, nlive))
    return np.concatenate((nk0, nk1))


def nk_live(nlive):
    return np.flip(np.arange(1, nlive + 1))


def logt_sample(n):
    """Generate logt for given number of live points n"""
    p = np.random.rand()
    return 1/n * np.log(p)
logt_sample = np.vectorize(logt_sample)

def X_sample(nk):
    """Generate single sample of X at the end of a live point distribution nk"""
    logt_samples = logt_sample(nk)
    logX = np.sum(logt_samples)
    X = np.exp(logX)
    return X

def generate_Xs(nk, iterations=None):
    """Generates the Xs at each iteration in a run with live point distribution nk"""
    logt_samples = logt_sample(nk)
    logXs = np.cumsum(logt_samples)
    Xs = np.exp(logXs)
    if iterations is not None:
        return np.take(Xs, iterations)
    return Xs

def generate_Xsamples(nk, n_samples=1000, iterations=None):
    """Repeats generate_Xs n_samples times and returns result as an array <n_samples> long"""
    if iterations is None:
        n_iterations = len(nk)
    else:
        n_iterations = len(iterations)
    X_samples = np.zeros((n_samples, n_iterations))
    for i in range(n_samples):
        X_samples[i] = generate_Xs(nk, iterations)    
    return X_samples

