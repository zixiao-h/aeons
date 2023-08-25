import numpy as np
import pandas as pd
from aeons.endpoint import EndModel
from aeons.regress import analytic_lm_params
from aeons.likelihoods import full


import numpy.polynomial.polynomial as poly
def linear_regress(x, y):
    coefs = poly.polyfit(x, y, deg=1)
    return coefs

def stopping_index(logZ_start, dlogZ_predict, epsilon=1e-3):
    logZ_predicted = logZ_start + dlogZ_predict.cumsum()
    logZ_tot = logZ_predicted[-1]
    logZ_f = np.log(1 - epsilon) + logZ_tot
    index_f = np.argmax([logZ_predicted > logZ_f])
    return index_f

class IncrementModel(EndModel):
    # Inherit from EndModel
    def __init__(self, samples, nlive=500):
        super().__init__(samples)
        self.nlive = nlive
    
    def true_delta_logZ(self, ndead):
        """True change in logZ every <nlive> ndeads from <ndead> onward"""
        logZs = self.logZs.iloc[ndead - self.nlive: len(self.samples): self.nlive]
        return logZs.diff(1).dropna()
    
    def delta_logZ(self, ndead, steps=10):
        if ndead < steps * self.nlive:
            raise ValueError("ndead must be greater than steps*nlive")
        logZs = self.logZs.iloc[:ndead:self.nlive]
        return logZs.diff(1).dropna().iloc[-steps:]
    
    def linear_extrapolate(self, ndead, steps=10):
        deltas = self.delta_logZ(ndead, steps)
        coefs = linear_regress(deltas.index.get_level_values(0), deltas)
        index_predict = np.arange(ndead - self.nlive, len(self.samples), self.nlive)
        dlogZ_predict = poly.polyval(index_predict, coefs)
        return pd.Series(index=index_predict[dlogZ_predict > 0], data=dlogZ_predict[dlogZ_predict > 0])
    
    def exponential_extrapolate(self, ndead, steps=10):
        nlive = self.nlive
        deltas = self.delta_logZ(ndead, steps)
        deltas_index = deltas.index.get_level_values(0).values
        x = np.linspace(0, 1, len(deltas))
        theta = analytic_lm_params(np.log(deltas.values), x, d0=1)
        index_predict = np.arange(ndead, 5*len(self.samples))
        x_predict = (index_predict - deltas_index[0]) / (deltas_index[-1] - deltas_index[0])
        dlogZ_predict = np.exp(full.func(x_predict, theta))
        return pd.Series(index=index_predict[::nlive], data=dlogZ_predict[::nlive])

    def get_endpoint(self, ndead, method='exp', steps=10):
        nlive = self.nlive
        if ndead < steps*nlive:
            raise ValueError("ndead must be greater than steps*nlive")
        if method == 'linear':
            delta_logZ_predict = self.linear_extrapolate(ndead, steps)
        elif method == 'exp':
            delta_logZ_predict = self.exponential_extrapolate(ndead, steps)
        else:
            raise ValueError("Method must be 'linear' or 'exp'")
        return ndead + (stopping_index(self.logZs.iloc[ndead], delta_logZ_predict.values) - 1) * nlive