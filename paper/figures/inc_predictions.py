import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
from aeons.increment import IncrementModel
figsettings()

name, samples = get_samples('gauss_16')
inc = IncrementModel(samples, nlive=500)
true_endpoint = inc.true_endpoint()

steps = 10
start = steps * 500/true_endpoint

iterations = make_iterations(true_endpoint, 50, start=start)
endpoints_exp = np.zeros(len(iterations))
endpoints_linear = np.zeros(len(iterations))
for i, ndead in enumerate(iterations):
    endpoints_exp[i] = inc.get_endpoint(ndead, method='exp', steps=steps)
    endpoints_linear[i] = inc.get_endpoint(ndead, method='linear', steps=2)

true_endpoint = endpoints[name]
ndeads, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/gauss_16.txt')
logXfs = np.array(logXfs)
logXs = samples.logX().iloc[ndeads]

true_logXf = samples.logX().iloc[true_endpoint]

fig, ax = plt.subplots(figsize=(3.5, 1.5))
ax.plot(-logXs, endpoints_exp/500, lw=.8, color='green', label='Exponential')
ax.plot(-logXs, endpoints_linear/500, lw=.8, color='orange', label='Linear')
plot_quantiles(-logXs, -logXfs, -true_logXf, ax=ax, label='Fit $\\mathcal{L}(X)$')

ax.set_ylim(0, 160)
ax.set_ylabel(r'$-\log \hat{X}_f$')
ax.set_xlabel(r'$-\log X$')
ax.legend(loc='upper right', fontsize=6)
ax.margins(x=0)

fig.savefig('inc_predictions.pdf', pad_inches=0, bbox_inches='tight')
