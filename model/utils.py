import torch.optim as optim
from pesq import pesq
from matplotlib import pyplot as plt
import soundfile as sf
import os
import numpy as np

'''
pesq: -0.5 to 4.5
'''
# def train(env, policy, optimizer, discount_factor):


#     loss = update_policy(returns, log_prob_actions, optimizer)
#     return loss, reward

# def update_policy(returns, log_prob_actions, optimizer):
#     returns = returns.detach()
#     loss = - (returns * log_prob_actions).sum()
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.item()

def reward_func(rate, ori, denoise):
    return pesq(rate, ori, denoise) * 100


def pesq(rate, ori, denoise):
    return pesq(rate, ori, denoise)


def custom_pesq(rate, ori, denoise):
    return 6 / (pesq(rate, ori, denoise) + 1.5)


def custom_pesq(soc):
    return 6 / (soc+1.5)


def fourier_bound(data, bound):
    return data[:bound]
    # if(data.shape[0]> bound):
    #     return data[:bound]
    # elif(data.shape[0]< bound):
    #     pad_array= np.zero[bound]
    #     pad_array[:data.shape[0]]= data
    #     return pad_array
    # else:
    #     return data


def show_plt(name, data):
    fig = plt.figure()
    plt.plot(data)
    plt.ylabel(name)
    plt.show


def save_flac(path, name, data, rate):
    sf.write(os.path.join(path, name), data, rate, format='FLAC')
