import numpy as np


class likelihood:
    def __init__(self, func, inverse, prime):
        self.func = func
        self.inverse = inverse
        self.prime = prime


def linear_like():
    def func(X, theta):
        return theta * X

    def inverse(y, theta, torch=True):
        return y / theta

    def prime(X, theta, torch=False):
        if torch:
            return theta * torch.ones_like(X)
        return theta * np.ones_like(X)

    return likelihood(func, inverse, prime)


def quad_like():
    def func(X, theta):
        return theta * X**2

    def inverse(y, theta, torch=True):
        return (y/theta)**(1/2)

    def prime(X, theta, torch=True):
        return 2 * theta * X
    
    return likelihood(func, inverse, prime)


def log_like():
    def func(X, theta):
        return theta * np.log(X)

    def inverse(y, theta, torch=False):
        if torch:
            return torch.exp(y/theta)
        return np.exp(y/theta)

    def prime(X, theta, torch=True):
        return theta / X

    return likelihood(func, inverse, prime)


def simple_like():
    def func(X, d):
        return -X**(2/d)

    def inverse(logL, d, torch=True):
        return (-logL)**(d/2)

    def prime(X, d, torch=True):
        return - (2/d) * X**(2/d - 1)

    return likelihood(func, inverse, prime)


def middle_like():
    def func(X, theta):
        d, sigma = theta
        return - X**(2/d)/(2*sigma**2)

    def inverse(logL, theta, torch=True):
        d, sigma = theta
        return (-2*sigma**2*logL)**(d/2)

    def prime(X, theta, torch=True):
        d, sigma = theta
        return - (1/d*sigma**2) * X**(2/d - 1)
    
    return likelihood(func, inverse, prime)


def full_like():
    def func(X, theta):
        logLmax, d, sigma = theta
        return logLmax - X**(2/d)/(2*sigma**2)

    def inverse(logL, theta, torch=True):
        logLmax, d, sigma = theta
        return (2*sigma**2 * (logLmax - logL))**(d/2)

    def prime(X, theta, torch=True):
        logLmax, d, sigma = theta
        return - 1/(d*sigma**2) * X**(2/d - 1)
    
    return likelihood(func, inverse, prime)

linear, quad, log = linear_like(), quad_like(), log_like()
simple, middle, full = simple_like(), middle_like(), full_like()