import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.increment import IncrementModel
figsettings()


name, samples = get_samples('toy', 'gauss_16')
inc = IncrementModel(samples, nlive=500)
true_endpoint = inc.true_endpoint()
true_logXf = samples.logX().iloc[true_endpoint]

ndead = 10000
endpoint_linear = inc.get_endpoint(ndead, method='linear', steps=2)
endpoint_exp = inc.get_endpoint(ndead, method='exp', steps=10)
logXf_linear = endpoint_linear/-500
logXf_exp = endpoint_exp/-500

fig, ax = plt.subplots(figsize=(2, 1.5))
inc.delta_logZ(ndead).plot(marker='x', ms=3, ls='none', ax=ax, label='Known $\\Delta\\log\\mathcal{Z}$')
inc.true_delta_logZ(ndead).plot(color='black', marker='x', ms=3, ls='none', ax=ax, label='True $\\Delta \\log \\mathcal{Z}$')
inc.exponential_extrapolate(ndead, steps=10).plot(lw=1, ax=ax, color='deeppink')
inc.linear_extrapolate(ndead, steps=2).plot(lw=1, ax=ax, color='deeppink')

ax.axvline(true_endpoint, lw=1, linestyle='--', color='black')
ax.axvline(x=endpoint_exp, lw=1, linestyle='--', color='deeppink')
ax.axvline(x=endpoint_linear, lw=1, linestyle='--', color='deeppink')
ax.set_xticks([endpoint_linear, true_endpoint, endpoint_exp], [-logXf_linear, f"{-true_logXf:.1f}", -logXf_exp])
height = 210
ax.text(endpoint_linear, height, "Linear", ha='center', va='top', fontsize=6)
ax.text(true_endpoint, height, "True", ha='center', va='top', fontsize=6)
ax.text(endpoint_exp, height, "Exponential", ha='center', va='top', fontsize=6)
ax.margins(y=0.01)

ax.set_xlim(ndead - 5000, true_endpoint * 2.5)
ax.set_xlabel("$-\\log X$")
ax.set_ylabel('$\\Delta \\log \\mathcal{Z}$')
ax.legend(loc='upper right', fontsize=6, handletextpad=.1)

fig.savefig('inc_extrapolate.pdf', pad_inches=0, bbox_inches='tight')
