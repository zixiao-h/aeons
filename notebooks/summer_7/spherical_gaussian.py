#%%
import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.endpoint import EndModel, theta_bandwidth_trunc
from aeons.plotting import *
figsettings()

# %%
from aeons.toy_samples import gaussian_samples
samples = gaussian_samples(500, 16, 0.01, seed=10)

# %%
model = EndModel(samples)
true_endpoint = model.true_endpoint()

# %%
samples = pickle_in('gauss/gauss_8.pkl')
model = EndModel(samples)
true_endpoint = model.true_endpoint()
iterations = make_iterations(true_endpoint, 50)
logXfs, logXfs_std = model.logXfs(theta_bandwidth_trunc, iterations, Nset=25, splits=1)

# %%
logXs = samples.logX().iloc[iterations]
endpoints, endpoints_std = calc_endpoints(iterations, logXs, logXfs, logXfs_std, nlive=500)
plot_std(iterations, endpoints, endpoints_std, true_endpoint, ylim=(0, 1.6))
plt.title(f"DKL = {samples.D_KL():.2f}")

#%%
pickle_dump('gauss_16.pkl', samples)
write_to_txt('gauss_16.txt', [iterations, logXfs, logXfs_std, true_endpoint])

# %%
ds = [8, 16, 32]
fig, axs = plt.subplots(1, 3, figsize=(8, 1.5))
for i, d in enumerate(ds):
    samples = pickle_in(f'gauss/gauss_{d}.pkl')
    iterations, logXfs, logXfs_std, true_endpoint = read_from_txt(f'gauss/gauss_{d}.txt')
    logXs = samples.logX().iloc[iterations]
    endpoints, endpoints_std = calc_endpoints(iterations, logXs, logXfs, logXfs_std, nlive=500)
    plot_std(iterations, endpoints, endpoints_std, true_endpoint, ylim=(0, 1.5), ax=axs[i])
    axs[i].set_title(f'$d={d}$')
fig.tight_layout()
