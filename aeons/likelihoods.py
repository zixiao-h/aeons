import numpy as np
import torch


class likelihood:
    def __init__(self, func, inverse, prime):
        self.func = func
        self.inverse = inverse
        self.prime = prime


def linear_like():
    def func(X, theta):
        return theta * X

    def inverse(y, theta, torched=True):
        return y / theta

    def prime(X, theta, torched=False):
        if torched:
            return theta * torch.ones_like(X)
        return theta * np.ones_like(X)

    return likelihood(func, inverse, prime)


def ax_b_like():
    def func(X, theta):
        a, b = theta
        return a * X + b

    def inverse(y, theta, torched=True):
        a, b = theta
        return (y - b)/a

    def prime(X, theta, torched=False):
        a, b = theta
        if torched:
            return a * torch.ones_like(X)
        return a * np.ones_like(X)

    return likelihood(func, inverse, prime)


def quad_like():
    def func(X, theta):
        return theta * X**2

    def inverse(y, theta, torched=True):
        return (y/theta)**(1/2)

    def prime(X, theta, torched=True):
        return 2 * theta * X
    
    return likelihood(func, inverse, prime)


def log_like():
    def func(X, theta):
        return theta * np.log(X)

    def inverse(y, theta, torched=False):
        if torched:
            return torch.exp(y/theta)
        return np.exp(y/theta)

    def prime(X, theta, torched=True):
        return theta / X

    return likelihood(func, inverse, prime)


def simple_like():
    def func(X, d, torched=True):
        return -X**(2/d)

    def inverse(logL, d, torched=True):
        return (-logL)**(d/2)

    def prime(X, d, torched=True):
        return -(2/d) * X**(-2/d - 1)

    return likelihood(func, inverse, prime)


def middle_like():
    def func(X, theta, torched=True):
        d, sigma = theta
        return - X**(2/d)/(2*sigma**2)

    def inverse(logL, theta, torched=True):
        d, sigma = theta
        return (-2*sigma**2*logL)**(d/2)

    def prime(X, theta, torched=True):
        d, sigma = theta
        return - (1/d*sigma**2) * X**(2/d - 1)
    
    return likelihood(func, inverse, prime)


def full_like():
    def func(X, theta, torched=True):
        logLmax, d, sigma = theta
        return logLmax - X**(2/d)/(2*sigma**2)

    def inverse(logL, theta, torched=True):
        logLmax, d, sigma = theta
        return (2*sigma**2 * (logLmax - logL))**(d/2)

    def prime(X, theta, torched=True):
        logLmax, d, sigma = theta
        return - 1/(d*sigma**2) * X**(2/d - 1)
    
    return likelihood(func, inverse, prime)

linear, quad, log = linear_like(), quad_like(), log_like()
simple, middle, full = simple_like(), middle_like(), full_like()
ax_b = ax_b_like()