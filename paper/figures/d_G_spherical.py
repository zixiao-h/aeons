import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.regress import *
from aeons.endpoint import *
from aeons.plotting import *
from aeons.beta import *
figsettings()

name, samples = get_samples('gauss_32')
iterations_grad, *d_Gs_grad = read_from_txt(f'{data_dir}/d_Gs/grad/{name}.txt')
iterations_logL, *d_Gs_logL = read_from_txt(f'{data_dir}/d_Gs/logL/{name}.txt')
iterations_post, *d_Gs_post = read_from_txt(f'{data_dir}/d_Gs/post/{name}.txt')
d_Gs_grad, d_Gs_logL, d_Gs_post = np.array(d_Gs_grad), np.array(d_Gs_logL), np.array(d_Gs_post)
logXs_grad = samples.logX().iloc[iterations_grad]
logXs_logL = samples.logX().iloc[iterations_logL]
logXs_post = samples.logX().iloc[iterations_post]

fig, ax = plt.subplots(figsize=(3.5, 1.3))
plot_quantiles(-logXs_post, d_Gs_post, samples.d_G(), ylim=(0, 1.5), ax=ax, color='red', label='Bayesian')
plot_quantiles(-logXs_logL, d_Gs_logL, samples.d_G(), ylim=(0, 1.5), ax=ax, color='orange', label='Canonical')
plot_quantiles(-logXs_grad, d_Gs_grad, samples.d_G(), ylim=(0, 1.5), ax=ax, color='black', label='Microcanonical')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], ncol=3, bbox_to_anchor=(1.05, -0.4))
ax.set_xlabel(r'$-\log X$')
ax.set_ylabel(r'$d_G$', rotation=0, labelpad=10)
ax.set_title('Intermediate dimension for spherical 32-d Gaussian')
fig.savefig('d_G_spherical.pdf', bbox_inches='tight', pad_inches=0)
