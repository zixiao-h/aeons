import numpy as np
import sympy as sp
import numba as nb

logLmax, d, sigma = sp.symbols('\log{L_\mathrm{max}} d \sigma')
X_i, logL_i, logL_j = sp.symbols('X_i \log{L_i} \log{L_j}')
mu_i, mu_j = sp.symbols('\mu_i, \mu_j')


def f(X_i, logLmax, d, sigma):
    return logLmax - X_i**(2/d)/(2*sigma**2)

def fprime(X_i, logLmax, d, sigma):
    return sp.diff(f(X_i, logLmax, d, sigma), X_i)

def abs_fprime(X_i, logLmax, d, sigma):
    return - fprime(X_i, logLmax, d, sigma)

def X_logLi(logL_i, logLmax, d, sigma):
    return (2*sigma**2 * (logLmax - logL_i))**(d/2)

def log_abs_fprime_Xstar(logL_i, logLmax, d, sigma):
    Xstar_i = X_logLi(logL_i, logLmax, d, sigma)
    return sp.log(abs_fprime(X_i, logLmax, d, sigma).subs(X_i, Xstar_i))

def cross_terms(logL_i, logL_j, logLmax, d, sigma):
    Xstar_i = X_logLi(logL_i, logLmax, d, sigma)
    Xstar_j = X_logLi(logL_j, logLmax, d, sigma)
    return (Xstar_i - mu_i) * (Xstar_j - mu_j)

def hess_fprime_i(theta_1, theta_2):
    symbolic_expr = sp.simplify(sp.diff(log_abs_fprime_Xstar(logL_i, logLmax, d, sigma), theta_1, theta_2))
    numeric_func = sp.lambdify([logL_i, logLmax, d, sigma], symbolic_expr)
    return numeric_func

def hess_cross_ij(theta_1, theta_2):
    symbolic_expr = sp.simplify(sp.diff(cross_terms(logL_i, logL_j, logLmax, d, sigma), theta_1, theta_2))
    numeric_func = sp.lambdify([logL_i, logL_j, mu_i, mu_j, logLmax, d, sigma], symbolic_expr)
    return numeric_func

def hess_logfprime(logL, theta_max, theta_1, theta_2):
    return hess_fprime_i(theta_1, theta_2)(logL, *theta_max).sum()

def hess_cross(logL, mean, covinv, theta_max, theta_1, theta_2):
    hess_tt_ij = hess_cross_ij(theta_1, theta_2)
    k = len(logL)
    @nb.jit
    def quad():
        quad = 0
        for i in range(k):
            cross_i = hess_tt_ij(logL[i], logL, mean[i], mean, *theta_max) * covinv[i]
            quad += cross_i.sum()
        return quad
    return quad()

def hess_tt(logL, mean, covinv, theta_max, theta_1, theta_2):
    return - hess_logfprime(logL, theta_max, theta_1, theta_2) - 1/2 * hess_cross(logL, mean, covinv, theta_max, theta_1, theta_2)


def hess(logL, mean, covinv, theta_max):
    dim = 3
    hess = np.zeros((dim, dim))
    hess[0][0] = hess_tt(logL, mean, covinv, theta_max, logLmax, logLmax)
    hess[0][1] = hess[1][0] = hess_tt(logL, mean, covinv, theta_max, logLmax, d)
    hess[0][2] = hess[2][0] = hess_tt(logL, mean, covinv, theta_max, logLmax, sigma)
    hess[1][1] = hess_tt(logL, mean, covinv, theta_max, d, d)
    hess[1][2] = hess[2][1] = hess_tt(logL, mean, covinv, theta_max, d, sigma)
    hess[2][2] = hess_tt(logL, mean, covinv, theta_max, sigma, sigma)
    return hess