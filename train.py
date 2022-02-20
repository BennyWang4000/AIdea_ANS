# %%
import yaml
import torch
from tqdm import tqdm
from math import floor
import torchvision.transforms as trns
import numpy as np

from model.dataset import AudioDataset
from model.models import UNet
from model.utils import show_plt, pesq_func, save_flac, fourier_bound, reward_func

from time import time

VERSION = 15

# %%
with open("config.yaml") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)

print(cfg)

transform = trns.Compose([
    trns.ToTensor()
])

dataset = AudioDataset(trns.ToTensor(), cfg['train_data_path'],
                       cfg['class'], cfg['signal_bound'])

train_num = floor(dataset.__len__() * cfg['train_per'])
print('train_num:\t', train_num)
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_num, dataset.__len__() - train_num])

dataloader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=cfg.get('batch_size'), shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:\t', device)
# %%
policy_net = UNet().to(device)
policy_net.train()
# policy_net = policy_net.float()
# target_net = UNet().to(device)
# summary(policy_net, (1, 2000))
# print(policy_net)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

loss_func = torch.nn.MSELoss()
# loss_func = F.mse_loss()

# g_params = list(netG_X2Y.parameters()) + list(netG_Y2X.parameters())
optimizer = torch.optim.Adam(
    policy_net.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))


loss_lst = []
pesq_lst = []
# %%


def main():
    is_start = 1
    for epoch in range(cfg['epochs']):
        print('epoch:\t', epoch)

        for data in tqdm(dataloader, total=len(dataloader)):
            ####################################################
            if is_start:
                t = time()

            ori, rate = data

            rate = rate[0].numpy()
            ori_f = np.real(
                np.fft.fft(ori, axis=1))

            ori_f_3 = np.expand_dims(ori_f, 1)
            # print('\t', ori_f_3.shape)  # (4,1,100000)

            if is_start:
                print('\n(1)\t', time() - t)
            ####################################################

            if is_start:
                t = time()

            noise_eval_f = np.squeeze(policy_net(
                torch.from_numpy(ori_f_3).float()))

            if is_start:
                print('(2)\t', time() - t)
            ####################################################

            if is_start:
                t = time()

            audio_pre_f = ori_f[:, :noise_eval_f.shape[1]
                                ] - noise_eval_f.detach().numpy()

            audio_pre = np.real(np.fft.ifft(audio_pre_f, axis=1))
            # audio_pre = np.fft.ifft(audio_pre_f.real).real
            # print('\n', type(rate), rate, '\n', type(ori), ori.shape,
            #       '\n', type(audio_pre), audio_pre.shape, '\n')

            if is_start:
                print('(3)\t', time() - t)
            ####################################################
            if is_start:
                t = time()

            pesq_soc = 0
            for i in range(ori.shape[0]):
                pesq_soc = pesq_soc + \
                    pesq_func(rate, ori.numpy()[i, :], audio_pre[i, :])

            if is_start:
                print('(4)\t', time() - t)
            ####################################################

            if is_start:
                t = time()

            reward = reward_func(pesq_soc)

            # loss = loss_func(torch.from_numpy(audio_pre_f),
            #                  torch.full(audio_pre_f.shape, reward), requires_grad=True)

            # loss = loss_func(noise_eval_f, reward +
            #                  cfg['gamma'] * noise_next_f.max(1)[0])
            # loss = loss_func(noise_eval_f, noise_eval_f + reward)
            # loss = policy_net.detach()-reward
            loss = noise_eval_f.mean()
            loss -= reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_start:
                print('(5)\t', time() - t)
            ####################################################
            pesq_lst.append(pesq_soc)
            loss_lst.append(loss)

            is_start = 0

        show_plt('loss', loss_lst)
        show_plt('pesq', pesq_lst)
        save_flac(cfg['saving_path'], str(epoch) +
                  '_pre.flac', audio_pre, rate)
        save_flac(cfg['saving_path'], str(epoch) + '_noise.flac',
                  np.fft.ifft(noise_eval_f.real, axis=1).real[0, :], rate)
        save_flac(cfg['saving_path'], str(epoch) + '_ori.flac', ori, rate)


# %%
if __name__ == '__main__':
    print('version:\t', VERSION)
    main()

# %%
