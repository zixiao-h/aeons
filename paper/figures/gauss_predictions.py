import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
figsettings()

d = [4, 8, 16, 32]
fig, axs = plt.subplots(1, 4, figsize=(7, 1.8))
axs = axs.flatten()
for i, nDims in enumerate(d):
    name, samples = get_samples(f'gauss_{nDims}')
    iterations, logXfs, logXfs_std, true_endpoint = read_from_txt(f'{aeons_dir}/data/predictions/dG_range/02_gauss_{nDims}.txt')
    logXf_true = samples.logX().iloc[int(true_endpoint[0])]
    logXs = samples.logX().iloc[iterations]
    # endpoints, endpoints_std = calc_endpoints(iterations, logXs, logXfs, logXfs_std, nlive=500)
    plot_std(-logXs, -logXfs, logXfs_std, -logXf_true, ylim=(0, 1.5), ax=axs[i])
    axs[i].set_title(rf'$d$ = {nDims}, $\mathcal{{D}}_\mathrm{{KL}} = {samples.D_KL():.1f}$', fontsize=8)
    axs[i].set_yticks([0, -logXf_true], [0, f"{-logXf_true:.1f}"])
fig.supxlabel('$-\\log X$', y=0.1, fontsize=8)
fig.supylabel('$-\\log \hat{X}_\mathrm{f}$',  fontsize=8)
fig.tight_layout()
fig.savefig('gauss_predictions.pdf', pad_inches=0, bbox_inches='tight')
