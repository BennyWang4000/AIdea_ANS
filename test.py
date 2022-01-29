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

with open("config.yaml") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

TRAIN_DATA_PATH = 'D://CodeRepositories//py_project//aidea//ANS//data//train'
SAMPLE_NAME = 'mixed_01001_cleaner.flac'
SAMPLE_NAME_2 = 'mixed_01002_siren.flac'

# %%
data, rate = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
data2, rate2 = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))

print(data.shape, data2.shape)
print(rate, rate2)
plt.figure(1)
plt.specgram(data)
plt.figure(2)
plt.plot(data)
plt.plot(data2)

print(pesq(rate, data, data2, 'wb'))
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
