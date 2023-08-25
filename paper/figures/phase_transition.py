import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
figsettings()

samples = pickle_in(f'{aeons_dir}/data/predictions/gauss/two.pkl')
iterations, logXfs, logXfs_std, true_endpoint = read_from_txt(f'{aeons_dir}/data/predictions/gauss/two_logXfs.txt')
iterations, d_Gs, d_Gs_std = read_from_txt(f'{aeons_dir}/data/predictions/gauss/two_d_Gs.txt')

logXf_true = samples.logX().iloc[int(true_endpoint[0])]
logXs = samples.logX().iloc[iterations]
endpoints, endpoints_std = calc_endpoints(iterations, logXs, logXfs, logXfs_std, nlive=705)

fig, axs = plt.subplots(1, 2, figsize=(4, 1.2))
plot_std(-logXs, -logXfs, logXfs_std, -logXf_true, ylim=(0.6, 1.1), color='deepskyblue', ax=axs[0])
plot_std(-logXs, d_Gs, d_Gs_std, d_Gs[-1], (0, 1.3), color='orange', ax=axs[1])
axs[0].set_title('$-\\log \hat{X_\mathrm{f}}$')
axs[1].set_title(f'$\\tilde{{d}}$')
fig.supxlabel('$-\\log X$', y=-0.2)
fig.savefig('phase_transition.pdf', pad_inches=0, bbox_inches='tight')
