import numpy as np
import matplotlib.pyplot as plt

proj_dir = '/home/zixiao/Documents/III/project/'
aeons_dir = '/home/zixiao/Documents/III/project/aeons'

def pickle_dump(filename, data):
    """Function that pickles data into a file"""
    import pickle
    pickle_out = open(filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_in(filename):
    import pickle
    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)
    return data


def write_to_txt(filename, data):
    try:
        open(filename+'.txt', 'w').close()
    except:
        print(f'creating {filename}.txt')
    with open(filename+'.txt', 'a') as f:
        for item in data:
            if not np.shape(item):
                item = [item]
            np.savetxt(f, np.array(item), newline=',')
            f.write('\n')

def read_from_txt(filename):
    data = []
    with open(filename+'.txt', 'r') as f:
        for line in f:
            data.append(np.fromstring(line.rstrip('\n'), sep=','))
    return data


chains = ['BAO', 'lensing', 'lensing_BAO', 'lensing_SH0ES', 'planck', 'planck_BAO', \
              'planck_lensing', 'planck_SH0ES', 'planck_lensing_BAO', 'planck_lensing_SH0ES', 'SH0ES']


def load_samples(which='lcdm'):
    samples_dict = {}
    for chain in chains:
        samples_dict[f'{which}_{chain}'] = pickle_in(f'{aeons_dir}/samples/{which}/{which}_{chain}.pickle')
    return samples_dict

def get_samples(which='lcdm', chain='BAO'):
    filename = f'{aeons_dir}/samples/{which}/{which}_{chain}.pickle'
    samples = pickle_in(filename)
    name = f'{which}_{chain}'
    return name, samples