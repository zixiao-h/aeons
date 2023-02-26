import numpy as np


class likelihood:
    def __init__(self, func, inverse, prime):
        self.func = func
        self.inverse = inverse
        self.prime = prime


def linear_like():
    def func(X, theta):
        return theta * X

    def inverse(y, theta):
        return y / theta

    def prime(X, theta):
        return theta * np.ones_like(X)

    return likelihood(func, inverse, prime)


def quad_like():
    def func(X, theta):
        return theta * X**2

    def inverse(y, theta):
        return (y/theta)**(1/2)

    def prime(X, theta):
        return 2 * theta * X
    
    return likelihood(func, inverse, prime)


def log_like():
    def func(X, theta):
        return theta * np.log(X)

    def inverse(y, theta):
        return np.exp(y/theta)

    def prime(X, theta):
        return theta / X

    return likelihood(func, inverse, prime)


def simple_like(logX=False):
    def func(X, d):
        if logX:
            X = np.exp(X)
        return -X**(2/d)

    def inverse(logL, d):
        if logX:
            return np.log((-logL)**(d/2))
        return (-logL)**(d/2)

    def prime(X, d):
        if logX:
            X = np.exp(X)
            return - (2/d) * X**(2/d)
        return - (2/d) * X**(2/d - 1)

    return likelihood(func, inverse, prime)


def middle_like(logX=False):
    def func(X, theta):
        d, sigma = theta
        if logX:
            X = np.exp(X)
        return - X**(2/d)/(2*sigma**2)

    def inverse(logL, theta):
        d, sigma = theta
        if logX:
            return np.log((-2*sigma**2*logL)**(d/2))
        return (-2*sigma**2*logL)**(d/2)

    def prime(X, theta):
        d, sigma = theta
        if logX:
            X = np.exp(X)
            return - (1/d*sigma**2) * X**(2/d)
        return - (1/d*sigma**2) * X**(2/d - 1)
    
    return likelihood(func, inverse, prime)


def full_like(logX=False):
    def func(X, theta):
        logLmax, d, sigma = theta
        if logX:
            X = np.exp(X)
        return logLmax - X**(2/d)/(2*sigma**2)

    def inverse(logL, theta):
        logLmax, d, sigma = theta
        if logX:
            return np.log((2*sigma**2 * (logLmax - logL))**(d/2))
        return (2*sigma**2 * (logLmax - logL))**(d/2)

    def prime(X, theta):
        logLmax, d, sigma = theta
        if logX:
            X = np.exp(X)
            return - 1/(d*sigma**2) * X**(2/d)
        return - 1/(d*sigma**2) * X**(2/d - 1)
    
    return likelihood(func, inverse, prime)