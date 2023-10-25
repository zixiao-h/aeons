import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
figsettings()


name, samples = get_samples('correlated_6d')
iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/{name}.txt')
true_endpoint = endpoints[name]
logXfs = np.array(logXfs)
logXs = samples.logX().iloc[iterations]
logXf_true = samples.logX().iloc[true_endpoint]
logXf_3d = get_samples('correlated_3d')[1].logX().iloc[endpoints['correlated_3d']]

fig, ax = plt.subplots(figsize=(3, 1.6))
plot_quantiles(-logXs, -logXfs, -logXf_true, ylim=(0, 1.2), color='deepskyblue', ax=ax)

ax.axhline(-logXf_3d, color='orange', linestyle='--', linewidth=.5)
ax.set_title('Elongated Gaussian', fontsize=8)
ax.set_yticks([0, -logXf_3d, -logXf_true], [0, r"$-\log X_\mathrm{f}^\mathrm{3d}$", r"$-\log X_\mathrm{f}^\mathrm{true}$"], rotation=0)
ax.set_xlabel('$-\\log X$')
ax.margins(x=0)

fig.savefig('elongated_logXfs.pdf', pad_inches=0, bbox_inches='tight')

