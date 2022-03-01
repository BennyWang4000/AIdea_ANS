# %%
# import wave
import torch
import librosa
import numpy as np
from tqdm.autonotebook import tqdm
from time import time
from math import floor
from torch.utils.data import DataLoader
from config import *
from model.models import UNet
from model.dataset import AudioDataset
from model.utils import pesq_func, reward_func, save_model, get_saving_path, mul_dim_stft, mul_dim_griffinlim, save_flac_via_lst, show_plt_via_lst


VERSION = 50
print('  version:\t', VERSION)


class SpeechEnhancement(object):
    def __init__(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.policy_net = UNet().to(DEV)
        # self.policy_net.double()

        self.loss_func = torch.nn.MSELoss()
        # self.loss_func = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR, betas=(BETA1, BETA2))

        # self.griffinlim = T.GriffinLim(
        #     n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEV)

        self.saving_path = get_saving_path(SAVING_PATH)

        self.train_loss_lst = []
        self.train_pesq_lst = []
        self.valid_loss_lst = []
        self.valid_pesq_lst = []

        self.ori_loss_lst = []
        self.reward_lst = []

    def get_batch_data(self, data):
        mixed, vocal, noise, rate = data
        mixed = mixed.float().to(DEV)
        mixed_np = mixed.clone().cpu().numpy()

        vocal = vocal.float().to(DEV)
        vocal_np = vocal.clone().cpu().numpy()

        noise = noise.float().to(DEV)
        noise_np = noise.cpu().numpy()

        rate_np = rate[0].numpy()

        return mixed, vocal, noise, mixed_np, vocal_np, noise_np, rate_np

    def loop(self, mixed_np, noise_np, batch_size):
        # ========================================================================================================

        mixed_stft_np = mul_dim_stft(mixed_np, batch_size)
        noise_stft_np = mul_dim_stft(noise_np, batch_size)
        noise_stft = torch.from_numpy(noise_stft_np).to(DEV)

        # ========================================================================================================

        noise_pre_stft = self.policy_net(
            torch.from_numpy(mixed_stft_np).float().to(DEV))
        noise_pre_stft_np = noise_pre_stft.clone().detach().cpu().numpy()

        # ========================================================================================================

        noise_pre_np = mul_dim_griffinlim(noise_pre_stft_np, batch_size)

        return noise_stft, noise_pre_np, noise_pre_stft

    def train(self):
        is_start = 1
        for epoch in range(EPOCHS):
            print('\nepoch:\t', epoch)
            train_bar = tqdm(self.train_loader)
            for n, data in enumerate(train_bar):
                self.policy_net.train()
                # ========================================================================================================

                mixed, vocal, noise, mixed_np, vocal_np, noise_np, rate_np = self.get_batch_data(
                    data)

                batch_size = mixed_np.shape[0]
                noise_stft, noise_pre_np, noise_pre_stft = self.loop(
                    mixed_np, noise_np, batch_size)

                # ========================================================================================================

                pesq_soc = 0.0

                noise_pre = torch.from_numpy(noise_pre_np).to(DEV)

                vocal_pre = mixed - noise_pre
                vocal_pre_np = vocal_pre.detach().cpu().numpy()

                try:
                    for i in range(batch_size):
                        pesq_soc = pesq_soc + pesq_func(rate_np, np.squeeze(vocal_np[i, :]),
                                                        np.squeeze(vocal_pre_np[i, :]))

                except Exception as ex:
                    print(type(ex), ex)
                    continue

                pesq_soc = pesq_soc / batch_size

                # ========================================================================================================
                # * update policy

                # ? reward

                # loss = loss_func(torch.from_numpy(vocal_pre_f),
                #                  torch.full(vocal_pre_f.shape, reward), requires_grad=True)
                # loss = loss_func(noise_pre_f, reward +
                #                  cfg['gamma'] * noise_next_f.max(1)[0])
                # loss = loss_func(noise_pre_f, noise_pre_f + reward)
                # loss = policy_net.detach()-reward

                # ? worked
                # loss = noise_pre.mean()
                # # loss = noise_pre_f.mean()
                # loss -= reward

                loss_v = self.loss_func(
                    vocal, vocal_pre)
                loss_n = self.loss_func(
                    noise, noise_pre)

                loss = self.loss_func(
                    noise_stft, noise_pre_stft)
                self.ori_loss_lst.append(loss.detach().cpu().numpy())

                loss = loss + torch.mean(loss_v + loss_n)

                reward = reward_func(pesq_soc)

                loss.backward()
                # loss += torch.autograd.Variable(torch.tensor([reward]))

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.train_pesq_lst.append(pesq_soc)
                self.train_loss_lst.append(loss.detach().cpu().numpy())
                self.reward_lst.append(reward)

                train_bar.set_postfix(
                    {'pesq': pesq_soc, 'reward': reward, 'loss': loss.detach().cpu().numpy()})

                # ========================================================================================================
                if n % 10 == 0:
                    show_plt_via_lst([
                        'train_loss_' + str(epoch) + '_' + str(n),
                        'train_pesq_' + str(epoch) + '_' + str(n),
                        'ori_loss_' + str(epoch) + '_' + str(n),
                        'reward_' + str(epoch) + '_' + str(n)],
                        [self.train_loss_lst,
                         self.train_pesq_lst,
                         self.ori_loss_lst,
                         self.reward_lst],
                        self.saving_path
                    )

                    save_model(self.policy_net.state_dict(),
                               self.saving_path, 'model.pt')

                    save_flac_via_lst([
                        'train_vocal_pre_' +
                        str(epoch) + '_' + str(n) + '.flac',
                        'train_mixed_ori_' +
                        str(epoch) + '_' + str(n) + '.flac',
                        'train_noise_pre_' +
                        str(epoch) + '_' + str(n) + '.flac'],
                        [np.squeeze(vocal_pre[0, :].detach().cpu().numpy()),
                         np.squeeze(mixed[0, :].detach().cpu().numpy()),
                         np.squeeze(noise_pre[0, :].detach().cpu().numpy())],
                        self.saving_path, rate_np
                    )

            if IS_VALID:
                valid_bar = tqdm(self.valid_loader)
                for n, data in enumerate(valid_bar):
                    self.policy_net.eval()
                    # ========================================================================================================

                    mixed, _, _, mixed_np, vocal_np, noise_np, rate_np = self.get_batch_data(
                        data)

                    batch_size = mixed_np.shape[0]
                    noise_stft, noise_pre_np, noise_pre_stft = self.loop(
                        mixed_np, noise_np, batch_size)

                    # ========================================================================================================

                    pesq_soc = 0.0
                    noise_pre = torch.from_numpy(noise_pre_np).to(DEV)
                    vocal_pre = mixed - noise_pre
                    vocal_pre_np = vocal_pre.detach().cpu().numpy()

                    try:
                        for i in range(batch_size):
                            pesq_soc = pesq_soc + pesq_func(rate_np, np.squeeze(vocal_np[i, :]),
                                                            np.squeeze(vocal_pre_np[i, :]))

                    except Exception as ex:
                        print(type(ex), ex)
                        continue

                    pesq_soc = pesq_soc / batch_size
                    # ========================================================================================================

                    loss = self.loss_func(
                        noise_stft, noise_pre_stft)

                    self.valid_pesq_lst.append(pesq_soc)
                    self.valid_loss_lst.append(loss.detach().cpu().numpy())

                    train_bar.set_postfix(
                        {'v_pesq': pesq_soc, 'v_loss': loss.detach().cpu().numpy()})

            show_plt_via_lst([
                'train_loss_' + str(epoch),
                'train_pesq_' + str(epoch),
                'valid_loss_' + str(epoch),
                'valid_pesq_' + str(epoch)],
                [self.train_loss_lst,
                    self.train_pesq_lst,
                    self.valid_loss_lst,
                    self.valid_pesq_lst],
                self.saving_path
            )

            save_model(self.policy_net.state_dict(),
                       self.saving_path, 'model.pt')

            save_flac_via_lst([
                'valid_vocal_pre' + str(epoch) + '.flac',
                'valid_mixed_ori' + str(epoch) + '.flac',
                'valid_noise_pre' + str(epoch) + '.flac'],
                [np.squeeze(vocal_pre[0, :].detach().cpu().numpy()),
                    np.squeeze(mixed[0, :].detach().cpu().numpy()),
                    np.squeeze(noise_pre[0, :].detach().cpu().numpy())],
                self.saving_path, rate_np
            )


# %%
if __name__ == '__main__':
    # with open("config.yaml") as fp:
    #     cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # print(cfg)

    dataset = AudioDataset(TRAIN_DATA_PATH, TRAIN_CLASS, SIGNAL_BOUND)
    train_num = floor(dataset.__len__() * TRAIN_PER)
    valid_num = dataset.__len__() - train_num
    print('   device:\t', DEV)
    print('train_num:\t', train_num)
    print('valid_num:\t', valid_num)

    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_num, valid_num])

    train_loader = DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

    # train_loader, valid_loader = AudioDataset.get_dataloader(train_set, val_set)

    se = SpeechEnhancement(train_loader=train_loader,
                           valid_loader=valid_loader)
    se.train()


# %%
