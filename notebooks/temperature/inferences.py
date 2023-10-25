import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.regress import *
from aeons.endpoint import *
from aeons.plotting import *
from aeons.beta import *
figsettings()

def inferences(d_G_method, points, ndead, Nset=25):
    d_G = d_G_method(points, ndead, Nset)[0]
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(d_G)
    logL, X_mean, nk, logZdead = data(points)
    logXf_set = np.zeros(Nset)
    ds = kde.resample()
    ds = ds[ds > 0]
    for i in range(Nset):
        X = generate_Xs(nk)
        d = np.random.choice(ds)
        theta = params_from_d(logL[ndead:], X[ndead:], d)
        logXf_set[i] = logXf_formula(theta, logZdead, X_mean[ndead])
    # logXf_set = logXf_set[~np.isnan(logXf_set)]
    # logXf_set = reject_outliers(logXf_set)
    return [logXf_set, d_G], f"{len(logXf_set)} samples, {d_G.mean():.1f}"

# Take in as command line argument name of chain
import sys
chain = sys.argv[1]
name, samples = get_samples(chain)
model = EndModel(samples)
true_endpoint = model.true_endpoint()
samples['beta_logL'] = get_betas_logL(samples)
print(f'Loaded chain {name}')

Nset = 25
iterations = make_iterations(true_endpoint, 50)
logXs = samples.logX().iloc[iterations]

logXfs_grad, d_Gs_grad = np.zeros((len(iterations), Nset)), np.zeros((len(iterations), Nset))
logXfs_logL, d_Gs_logL = np.zeros((len(iterations), Nset)), np.zeros((len(iterations), Nset))
logXfs_post, d_Gs_post = np.zeros((len(iterations), Nset)), np.zeros((len(iterations), Nset))
for i, ndead in enumerate(iterations):
    points = points_at_iteration(samples, ndead)
    logXfs_grad[i], d_Gs_grad[i] = inferences(get_d_G_grad, points, ndead, Nset)[0]
    logXfs_logL[i], d_Gs_logL[i] = inferences(get_d_G_logL, points, ndead, Nset)[0]
    logXfs_post[i], d_Gs_post[i] = inferences(get_d_G_post, points, ndead, Nset)[0]
    print('\r', f"Iteration {ndead} of {iterations[-1]}", end='')


write_to_txt(f'{data_dir}/logXfs/grad/{name}.txt', [iterations, *logXfs_grad])
write_to_txt(f'{data_dir}/logXfs/logL/{name}.txt', [iterations, *logXfs_logL])
write_to_txt(f'{data_dir}/logXfs/post/{name}.txt', [iterations, *logXfs_post])
write_to_txt(f'{data_dir}/d_Gs/grad/{name}.txt', [iterations, *d_Gs_grad])
write_to_txt(f'{data_dir}/d_Gs/logL/{name}.txt', [iterations, *d_Gs_logL])
write_to_txt(f'{data_dir}/d_Gs/post/{name}.txt', [iterations, *d_Gs_post])

print('Wrote d_Gs and logXfs to data/d_Gs and data/logXfs')
