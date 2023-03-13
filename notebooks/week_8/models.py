import numpy as np
import torch
from scipy.optimize import minimize
from torch.autograd.functional import hessian


def logZ(logPmax, H, D, details=False):
    if details:
        print(f"logPr_max: {logPmax}, Hessian: {- 1/2 * np.log(abs(np.linalg.det(H)))}")
    return logPmax - 1/2 * np.log(abs(np.linalg.det(H))) + D/2 * np.log(2*np.pi)


class Model:
    def __init__(self, y, likelihood, mean):
        self.y = y
        self.likelihood = likelihood
        self.mean = mean
        self.N = len(y)

    def minimise(self, x0, bounds=None, full_solution=False):
        def func(theta):
            return - self.logPr(theta)
        solution = minimize(func, x0, method='Nelder-Mead', bounds=bounds)
        if full_solution:
            return solution
        else:
            return solution.x
    
    def logZ(self, theta_max, details=False):
        logPr_max = self.logPr(theta_max)
        H = self.hess(theta_max)
        D = len(theta_max) + 1
        return logZ(logPr_max, H, D, details)
    

class LS(Model):
    def __init__(self, y, likelihood, mean):
        super().__init__(y, likelihood, mean)
    
    def L_sq(self, theta):
        loss = self.mean - self.likelihood.inverse(self.y, theta)
        return np.sum(loss**2)
    
    def s(self, theta):
        return np.sqrt(self.L_sq(theta)/self.N)

    def logPr(self, theta):
        L_sq = self.L_sq(theta)
        s = np.sqrt(L_sq/self.N)
        return -1/2 * self.N * np.log(2*np.pi*s**2) - L_sq/(2*s**2)
    
    def hess(self, theta_max):
        s = self.s(theta_max)
        y = torch.from_numpy(self.y)
        mean = torch.from_numpy(self.mean)
        theta_s_max = torch.tensor([*theta_max, s], requires_grad=True)
        def func(theta_s):
            if len(theta_s) == 2:
                theta, s = theta_s
            else:
                *theta, s = theta_s
            loss = mean - self.likelihood.inverse(y, theta, torched=True)
            L_sq = torch.sum(loss**2)
            return -1/2 * self.N * torch.log(2*torch.pi*s**2) - L_sq/(2*s**2)
        H = hessian(func, theta_s_max)
        return np.array(H)
    

class CG(Model):
    def __init__(self, y, likelihood, mean, covinv):
        super().__init__(y, likelihood, mean)
        self.covinv = covinv
        self.logdet_inv = np.linalg.slogdet(covinv)[1]
    
    def logPr(self, theta):
        Xstar = self.likelihood.inverse(self.y, theta)
        log_abs_fprimes = np.log(abs(self.likelihood.prime(Xstar, theta)))
        return 1/2 * self.logdet_inv - np.sum(log_abs_fprimes) - 1/2 * (Xstar - self.mean).T @ self.covinv @ (Xstar - self.mean)
    
    def hess(self, theta_max):
        y = torch.from_numpy(self.y)
        mean = torch.from_numpy(self.mean)
        covinv = torch.from_numpy(self.covinv)
        theta_max = torch.tensor(theta_max, requires_grad=True)
        def func(theta):
            Xstar = self.likelihood.inverse(y, theta, torched=True)
            log_abs_fprimes = torch.log(abs(self.likelihood.prime(Xstar, theta, torched=True)))
            return - torch.sum(log_abs_fprimes) - 1/2 * (Xstar - mean).T @ covinv @ (Xstar - mean)
        H = hessian(func, theta_max)
        return np.array(H)