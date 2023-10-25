import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.regress import *
from aeons.endpoint import *
from aeons.plotting import *
from aeons.beta import *
figsettings()

import sys
chain = sys.argv[1]
name, samples = get_samples(chain)
model = EndModel(samples)
true_endpoint = model.true_endpoint()
true_logXf = model.true_logXf()

iterations = make_iterations(true_endpoint, 50)
logXs = samples.logX().iloc[iterations]
logXfs, d_Gs = model.inferences(get_d_G_post, iterations, Nset=50)

write_to_txt(f'{data_dir}/logXfs/post/{name}.txt', [iterations, *logXfs])
write_to_txt(f'{data_dir}/d_Gs/post/{name}.txt', [iterations, *d_Gs])
print('Written to data/logXfs and data/d_Gs with 50 iterations')

fig, axs = plt.subplots(1, 2, figsize=(8, 2))
plot_quantiles(-logXs, d_Gs, samples.d_G(), (0, 1.5), ax=axs[0], color='lightseagreen')
plot_quantiles(-logXs, -logXfs, -true_logXf, (0, 1.5), ax=axs[1])
fig.suptitle(f'{name}')