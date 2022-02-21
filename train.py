# %%
import yaml
import torch
from tqdm import tqdm
from math import floor
import torchvision.transforms as trns
import numpy as np

from model.dataset import AudioDataset
from model.models import UNet
from model.utils import show_plt, pesq_func, save_flac, save_model, reward_func
from model.mixit_wrapper import MixITLossWrapper

from time import time

VERSION = 29

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

loss_func = MixITLossWrapper(torch.nn.MSELoss())
# loss_func = torch.nn.MSELoss()
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

            noise_eval_f_3 = policy_net(
                torch.from_numpy(ori_f_3).float().to(device))

            noise_eval_f = np.squeeze(noise_eval_f_3)

            if is_start:
                print('(2)\t', time() - t)
            ####################################################

            if is_start:
                t = time()

            audio_pre_f = ori_f - noise_eval_f.detach().cpu().numpy()
            audio_pre_f_3 = ori_f_3 - noise_eval_f_3.detach().cpu().numpy()

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

            try:
                for i in range(ori.shape[0]):
                    pesq_soc = pesq_soc + pesq_func(
                        rate, ori.cpu().numpy()[i, :], audio_pre[i, :]
                    )
            except:
                print('error')
                continue

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

            # ? worked
            # loss = noise_eval_f.mean()
            # loss -= reward

            print('\n')
            print(audio_pre_f.shape, ori_f.shape)
            print('\n')

            loss = loss_func(audio_pre_f_3, ori_f_3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_start:
                print('(5)\t', time() - t)
            ####################################################
            pesq_lst.append(pesq_soc)
            loss_lst.append(loss.detach().cpu().numpy())

            is_start = 0

        save_ori = np.squeeze(ori[0, :])

        show_plt('loss', loss_lst, cfg['saving_path'])
        show_plt('pesq', pesq_lst, cfg['saving_path'])
        save_flac(cfg['saving_path'], str(epoch) +
                  '_pre.flac', np.squeeze(audio_pre[0, :save_ori.shape[0]]), rate)
        save_flac(cfg['saving_path'], str(epoch) +
                  '_ori.flac', save_ori, rate)
        save_flac(cfg['saving_path'], str(epoch) + '_noise.flac',
                  np.real(np.fft.ifft(np.squeeze(noise_eval_f.detach().cpu().numpy()[0, :]))), rate)

        save_model(cfg, cfg['saving_path'], 'model.pt')


# %%
if __name__ == '__main__':
    print('version:\t', VERSION)
    main()

# %%
