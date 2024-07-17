import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
figsettings()

import os
dir_path = os.path.dirname( os.path.realpath( __file__ ) )

name, samples = get_samples('correlated_6d')
iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/{name}.txt')
true_endpoint = endpoints[name]
logXfs = np.array(logXfs)
logXs = samples.logX().iloc[iterations]
logXf_true = samples.logX().iloc[true_endpoint]
logXf_3d = get_samples('correlated_3d')[1].logX().iloc[endpoints['correlated_3d']]

fig, ax = plt.subplots(figsize=(3, 1.6))
plot_quantiles(-logXs, -logXfs, -logXf_true, ylim=(0, 1.2), color='deepskyblue', ax=ax)

ax.axhline(-logXf_true, color='red', linestyle='--', linewidth=.5, label='$d=6$ endpoint (truth)')
ax.axhline(-logXf_3d, color='orange', linestyle='--', linewidth=.5, label='$d=3$ endpoint')

ax.set_yticks([0, -logXf_3d, -logXf_true], [0, f"{-logXf_3d:.1f}", f"{-logXf_true:.1f}"], rotation=0)
ax.set_title('Elongated Gaussian', fontsize=8)
ax.set_xlabel('$-\\log X$')
ax.set_ylabel('$-\\log \\hat{X}_\\mathrm{f}$')
ax.margins(x=0)
ax.legend()

fig.savefig(dir_path + '/' + 'elongated_logXfs.pdf', pad_inches=0, bbox_inches='tight')

