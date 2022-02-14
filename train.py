# %%
from email.mime import audio
from pickletools import optimize
import yaml
import torch
from tqdm import tqdm
import speechbrain as sb
from math import floor

import numpy as np

from torchsummary import summary
from model.dataset import AudioDataset
from model.models import Generator, Discriminator, Model
from model.utils import custom_pesq, show_plt, pesq, save_flac, fourier_bound, reward_func

VERSION = 8

# %%
with open("config.yaml") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)

print(cfg)
dataset = AudioDataset(cfg['train_data_path'], cfg['class'])

train_num = floor(dataset.__len__() * cfg['train_per'])
print('train_num:\t', train_num)
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_num, dataset.__len__() - train_num])

dataloader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=cfg.get('batch_size'), shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:\t', device)
# %%
net = Model().to(device)
# print(net)
print(summary(net, (200000,)))
# netG_X2Y = Generator().to(device)
# netG_Y2X = Generator().to(device)
# netD_X = Discriminator().to(device)
# netD_Y = Discriminator().to(device)

loss_func = torch.nn.MSELoss().to(device)


# g_params = list(netG_X2Y.parameters()) + list(netG_Y2X.parameters())
optimizer = torch.optim.Adam(
    net.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))

# # Create optimizers for the generators and discriminators
# g_optimizer = torch.optim.Adam(
#     g_params, lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))

# d_x_optimizer = torch.optim.Adam(
#     netD_X.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
# d_y_optimizer = torch.optim.Adam(
#     netD_Y.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))

loss_lst = []
pesq_lst = []
# %%


def main():
    for epoch in range(cfg['epochs']):
        print('epoch: ', epoch)

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, data in enumerate(progress_bar):
            if (i % 100 == 0):
                print(i, '...')

            print(data)
            ori = data['audio'].to(device)
            rate = data['data'].to(device)
            ori_f = fourier_bound(np.fft.fft(ori), cfg['signal_bound'])

            optimizer.zero_grad()

            noise_pre_f = net(ori_f)
            audio_pre_f = ori_f - noise_pre_f

            audio_pre = np.fft.ifft(audio_pre_f.real).real
            pesq_soc = pesq(rate, ori, audio_pre)
            reward = reward_func(pesq_soc)

            loss = loss_func(ori, noise_pre_f) * reward
            loss.backward()

            pesq_lst.append(pesq_soc)
            loss_lst.append(loss)

            optimizer.step()

            # ? CYCLE GAN
            # audio_X2Y = netG_X2Y(audio_data)
            # audio_Y2X = netG_Y2X(audio_data)

            # loss_X2Y = loss_func(audio_data, audio_X2Y)
            # loss_Y2X = loss_func(audio_data, audio_Y2X)

        show_plt('loss', loss_lst)
        show_plt('pesq', pesq_lst)
        save_flac(cfg['saving_path'], str(epoch) +
                  '_pre.flac', audio_pre, rate)
        save_flac(cfg['saving_path'], str(epoch) + '_noise.flac',
                  np.fft.ifft(noise_pre_f.real).real, rate)
        save_flac(cfg['saving_path'], str(epoch) + '_ori.flac', ori, rate)
# %%


if __name__ == '__main__':
    print('version:\t', VERSION)
    main()
