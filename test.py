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
import numpy as np
# import scipy.io.wavfile

from model.models import Model
from torchsummary import summary

with open("config.yaml") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

TRAIN_DATA_PATH = 'D://CodeRepositories//py_project//aidea//ANS//data//train'
SAVING_PATH = 'C://Users//costco//Desktop'
SAMPLE_NAME = 'mixed_01001_cleaner.flac'
SAMPLE_NAME_2 = 'mixed_01603_train.flac'
#%%

X = np.random.uniform(-10, 10, 70).reshape(-1, 7)
print(X)
#%%

model= Model()

print(summary(model, (200000,)))
# %%
# ? audio signal
data, rate = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
data2, rate2 = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))
# rate, data  = scipy.io.wavfile.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
# rate2, data2  = scipy.io.wavfile.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))

print(data.shape, data2.shape)
print(rate, rate2)
plt.figure(1)
plt.specgram(data)
plt.figure(2)
plt.plot(data)
plt.plot(data2)
print(pesq(rate, data, data2, 'wb'))
# %%
# ? fourier
data1_fourier = np.fft.fft(data, )
data2_fourier = np.fft.fft(data2)
print(data2_fourier.shape)
print(data2_fourier.shape[0])
print(data2_fourier.shape[1])
print(data2_fourier.shape[2])
plt.figure(3)
plt.plot(data2_fourier)
plt.plot(data1_fourier)
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

print(config_params.get('epochs'))
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# %%
print(config_params.get('train_data_path'))
train_dataset = AudioDataset(config_params.get('train_data_path'))

dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=config_params.get('batch_size'), shuffle=True)

for epoch in range(config_params.get('epochs')):
    # progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, re in enumerate(dataloader):
        pass
print('f')
# %%
root = config_params.get('train_data_path')
ist = os.listdir(root)

print(ist)

# %%
