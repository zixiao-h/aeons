import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
figsettings()

fig, axs = plt.subplots(1, 4, figsize=(7, 1.6))
for i, exp in enumerate([2, 3, 4, 5]):
    ax = axs[i]
    name, samples = get_samples(f'cauchy_8{exp}')
    iterations, logXfs, logXfs_std, true_endpoint = read_from_txt(f'{aeons_dir}/data/predictions/dG_range/02_cauchy_8{exp}.txt')
    logXs = samples.logX().iloc[iterations]
    logXf_true = samples.logX().iloc[int(true_endpoint[0])]
    plot_std(-logXs, -logXfs, logXfs_std, -logXf_true, ylim=(0, 2), color='deepskyblue', ax=ax)
    ax.set_title(f'$\\gamma=10^{{{-exp}}}$, $\\mathcal{{D}}_\\mathrm{{KL}} = {samples.D_KL():.1f}$', fontsize=8)
    ax.set_yticks([0, -logXf_true], [0, f"{-logXf_true:.1f}"])

fig.supxlabel('$-\\log X$', y=0.02)
fig.supylabel('$-\\log \\hat{X_\\mathrm{f}}$', x=0.01)
fig.tight_layout()
fig.savefig('cauchy_predictions.pdf', pad_inches=0, bbox_inches='tight')
