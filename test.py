# %%
import glob
import soundfile as sf
from pesq import pesq
import os
import yaml
import torch
from model.dataset import AudioDataset
from matplotlib import pyplot as plt
import tqdm
import scipy.fft
from math import floor
import numpy as np
import torchvision.transforms as trns
# import scipy.io.wavfile

from model.models import UNet
from torchsummary import summary

with open("config.yaml") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)

TRAIN_DATA_PATH = 'D://CodeRepositories//py_project//aidea//ANS//data//train'
SAVING_PATH = 'C://Users//costco//Desktop'
SAMPLE_NAME = 'mixed_01001_cleaner.flac'
SAMPLE_NAME_2 = 'mixed_16605_train.flac'
print('done')
# %%
z = np.zeros((5,))
a = np.array([1, 2, 3])
print(z.shape)
print(a.shape)
z[:a.shape[0]] = a
print(z)
# %%
o = np.ones((1, 5))
np.expand_dims(o, axis=0)
print(o)
print(o.shape)
o = np.squeeze(o)
print(o)
print(o.shape)

oo = np.ones((2, 1, 5))
print(oo)
print(oo.shape)
on = np.squeeze(oo, 1)
print(on)
print(on.shape)

# %%
#? model
# enc= Encoder()
# dec= Decoder()
# print(enc)
# print('===========')
# print(dec)
model = UNet()

print(summary(model, (1, 90000), batch_size=4))
# print(model)
# %%
# ? audio signal
data, rate = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
data2, rate2 = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))
# rate, data  = scipy.io.wavfile.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
# rate2, data2  = scipy.io.wavfile.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))
# sf.write(os.path.join(SAVING_PATH, 'zzori.flac'),
#          data, rate, format='FLAC')
zz = np.zeros((300000,))
zz[:data.shape[0]] = data
print(data.shape, data2.shape)
print(rate, rate2)
plt.figure(1)
plt.specgram(data)
# sf.write(os.path.join(SAVING_PATH, 'zz.flac'),
#          zz, rate, format='FLAC')
print(zz.shape, ', ', data.shape)

plt.figure(2)
plt.plot(data)
plt.plot(data2)
print(pesq(rate, data, data2, 'wb'))
print(pesq(rate, zz, data2, 'wb'))

# %%
# ? fourier
data1_fourier = np.fft.fft(data, )
data2_fourier = np.fft.fft(data2)
print(data2_fourier.shape)
print(data2.shape)
# print(data2_fourier.shape[0])
plt.figure(3)
plt.plot(data2_fourier)
plt.plot(data2)
# %%
sq = np.expand_dims(data1_fourier, axis=0)
print(sq.shape)


# %%
data2_10w = data2_fourier[:100000]
data2_10w = data2_10w
print('!!', data2_10w.dtype)
print(data2_10w.shape)
print(type(data2_10w[0]))
print((data2_10w[0]))
sf.write(os.path.join(SAVING_PATH, '10w.flac'),
         np.fft.ifft(data2_10w), rate, format='FLAC')

data2_2w = data2_fourier[:20000]
data2_2w = data2_2w.real
print(data2_2w.shape)
print(type(data2_2w[0]))
print((data2_2w[0]))
sf.write(os.path.join(SAVING_PATH, '2w.flac'),
         np.fft.ifft(data2_2w).real, rate, format='FLAC')

data2_1w = data2_fourier[:200000]
data2_1w = data2_1w.real
print(data2_1w.shape)
print(type(data2_1w[0]))
print((data2_1w[0]))
sf.write(os.path.join(SAVING_PATH, '1w.flac'),
         np.fft.ifft(data2_1w).real, rate, format='FLAC')

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# %%
transform = trns.Compose(
    trns.ToTensor
)

print(cfg.get('train_data_path'))
train_dataset = AudioDataset(transform, cfg['train_data_path'],
                             cfg['class'], cfg['signal_bound'])

dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=cfg.get('batch_size'), shuffle=True)

for epoch in range(cfg.get('epochs')):
    # progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in enumerate(dataloader):
        print(data)
        print(type(data))

        break
print('f')
# %%

dataset = AudioDataset(transform, cfg['train_data_path'],
                       cfg['class'], cfg['signal_bound'])

train_num = floor(dataset.__len__() * cfg['train_per'])
print('train_num:\t', train_num)
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_num, dataset.__len__() - train_num])

dataloader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=cfg.get('batch_size'), shuffle=True)


# %%
root = cfg.get('train_data_path')
ist = os.listdir(root)

print(ist)

# %%
