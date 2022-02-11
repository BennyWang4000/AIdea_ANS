import torch.optim as optim
from pesq import pesq
from matplotlib import pyplot as plt
import soundfile as sf
import os

'''
pesq: -0.5 to 4.5
'''


def pesq(rate, ori, denoise):
    return pesq(rate, ori, denoise)


def custom_pesq(rate, ori, denoise):
    return 6 / (pesq(rate, ori, denoise) + 1.5)


def custom_pesq(soc):
    return 6 / (soc+1.5)


def show_plt(name, data):
    fig = plt.figure()
    plt.plot(data)
    plt.ylabel(name)
    plt.show


def save_flac(path, name, data, rate):
    sf.write(os.path.join(path, name), data, rate, format='FLAC')
