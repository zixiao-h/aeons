import matplotlib.pyplot as plt
from util import read_from_txt, figsettings, plot_std
figsettings()

fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.3))

iterations, logbetas_grad, logbetas_logL, logbetas_mean, logbetas_std = read_from_txt('data/logbetas/gauss_32.txt')
logXs = read_from_txt('data/logXs/gauss_32.txt')[0][iterations.astype(int)]
plot_std(-logXs, logbetas_mean, logbetas_std, label=f'Bayesian', ax=axs[0], color='red')
axs[0].plot(-logXs, logbetas_logL, color='orange', label='Canonical')
axs[0].plot(-logXs, logbetas_grad, color='black', lw=2, label='Microcanonical')

iterations, logbetas_grad, logbetas_logL, logbetas_mean, logbetas_std = read_from_txt(f'data/logbetas/slab_spike.txt')
logXs = read_from_txt('data/logXs/slab_spike.txt')[0][iterations.astype(int)]
plot_std(-logXs, logbetas_mean, logbetas_std, label=f'Bayesian', ax=axs[1], color='red')
axs[1].plot(-logXs, logbetas_logL, color='orange', label='Canonical')
axs[1].plot(-logXs, logbetas_grad, color='black', lw=2, label='Microcanonical')

for ax in axs:
    ax.set_xlabel(r"$-\log X$")

axs[0].set_ylabel(r'$\log \beta$')    
axs[1].set_ylim(-8, 8)

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles[::-1], labels[::-1], ncol=3, bbox_to_anchor=(2.2, -0.4), fontsize=7)
axs[0].set_title('Spherical Gaussian', fontsize=7)
axs[1].set_title('Slab-spike Gaussian', fontsize=7)
fig.suptitle('Temperature vs compression for different distributions', y=1.15, fontsize=8)
fig.savefig('beta_comparison.pdf', bbox_inches='tight', pad_inches=0)
