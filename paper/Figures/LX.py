import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import figsettings

figsettings()


def full(X, theta):
    logLmax, d, sigma = theta
    return logLmax - X**(2/d)/(2*sigma**2)

logXarray = np.linspace(-70, 0, 1000)
Xarray = np.exp(logXarray)
theta = [0, 10, 0.01]
logL = full(Xarray, theta)
L = np.exp(logL)

Ztrue = np.trapz(L, Xarray)
LlogL = np.nan_to_num(L/Ztrue * np.log(L/Ztrue))
H = np.trapz(LlogL, Xarray)

fig, ax1 = plt.subplots(figsize=(3.37, 1.5))
ax2 = ax1.twinx()
ax1.plot(logXarray, L, color='black', label=r"$L(X)$")
ax2.fill(logXarray, L*Xarray, lw=1, color='black', alpha=0.2, label=r"$L(X)X$")
ax2.axvline(-H, ls='--', lw=0.5, color='black')
for ax in ax1, ax2:
    ax.set_yticks([])
    ax.set_xticks([-H], [f'$-D_\mathrm{{KL}}$'])
    ax.margins(x=0, y=0)
ax1.set_xlabel(r'$\log X$');
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
fig.tight_layout()
fig.savefig('LX.pdf')
