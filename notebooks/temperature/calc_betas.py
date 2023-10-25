# Calculates microcanonical, canonical and Bayesian temperatures and writes them to file
import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.regress import *
from aeons.endpoint import *
from aeons.plotting import *
from aeons.beta import *
figsettings()

name, samples = get_samples('gauss_32')
model = EndModel(samples)
true_endpoint = model.true_endpoint()
true_logXf = model.true_logXf()

iterations = make_iterations(true_endpoint, 100)
logXs = samples.logX().iloc[iterations]
logbetas_mean, logbetas_std = np.ones(len(iterations)), np.ones(len(iterations))
logbetas_grad = np.ones(len(iterations))
for i, ndead in enumerate(iterations):
    points = points_at_iteration(samples, ndead)
    logbetas_mean[i], logbetas_std[i] = get_logbeta_post(points, ndead)
    logbetas_grad[i] = get_logbeta_grad(points, ndead)
    print('\r', i, end='')
logbetas_logL = np.log(get_betas_logL(samples)[iterations])
# write_to_txt(f'{data_dir}/logbetas/{name}.txt', [iterations, logbetas_grad, logbetas_logL, logbetas_mean, logbetas_std])