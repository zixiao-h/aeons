import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
figsettings()

fig, axs = plt.subplots(1, 3, figsize=(8, 1.4))
for i, exp in enumerate([2, 3, 4]):
    ax = axs[i]
    name, samples = get_samples('toy', f'cauchy_8{exp}')
    iterations, logXfs, logXfs_std, true_endpoint = read_from_txt(f'{aeons_dir}/data/predictions/cauchy/cauchy_8{exp}.txt')
    logXs = samples.logX().iloc[iterations]
    logXf_true = samples.logX().iloc[int(true_endpoint[0])]
    plot_std(-logXs, -logXfs, logXfs_std, -logXf_true, ylim=(0, 1.6), color='deepskyblue', ax=ax)
    ax.set_title(f'$\\gamma=10^{{{-exp}}}$, $\\mathcal{{D}}_\\mathrm{{KL}} = {samples.D_KL():.1f}$', fontsize=8)

fig.supxlabel('$-\\log X$', y=-0.15)
fig.supylabel('$-\\log \\hat{X_\\mathrm{f}}$', x=0.06)
fig.savefig('cauchy_predictions.pdf', pad_inches=0, bbox_inches='tight')
