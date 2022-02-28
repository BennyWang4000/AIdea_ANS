# %%
# import wave
import torch
import librosa
import numpy as np
from tqdm.autonotebook import tqdm
from time import time
from math import floor
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
from model.models import UNet
from model.dataset import AudioDataset
from model.mixit_wrapper import MixITLossWrapper
from model.utils import show_plt, pesq_func, save_flac, save_model, reward_func, get_saving_path


VERSION = 10
print('  version:\t', VERSION)


class SpeechEnhancement(object):
    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.policy_net = UNet().to(DEV)
        # self.policy_net.double()
        self.policy_net.train()

        self.loss_func = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR, betas=(BETA1, BETA2))

        # self.griffinlim = T.GriffinLim(
        #     n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEV)

        self.saving_path = get_saving_path(SAVING_PATH)

        self.loss_lst = []
        self.pesq_lst = []

    def train(self):
        is_start = 1
        for epoch in range(EPOCHS):
            print('\nepoch:\t', epoch)
            for n, data in enumerate(tqdm(self.train_loader)):
                ####################################################
                if is_start:
                    t = time()

                mixed, vocal, noise, rate = data
                # * torch.float64
                mixed = mixed.float().to(DEV)
                mixed_np = mixed.clone().cpu().numpy()

                vocal = vocal.float().to(DEV)
                vocal_np = vocal.clone().cpu().numpy()

                noise = noise.float().to(DEV)
                noise_np = noise.cpu().numpy()

                rate_np = rate[0].numpy()

                # mixed_stft = torch.abs(torch.stft(
                #     mixed, n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=False).sum(-1)).to(DEV)

                # mixed_stft_mag= torch.abs(
                #     torch.sqrt(
                #         torch.pow(mixed_stft[:, : , 0], 2)+
                #         torch.pow(mixed_stft[:, :, 1], 2)
                #     )
                # )

                # ? librosa
                mixed_stft_np = np.empty(
                    (mixed.shape[0], STFT_SHAPE[0], STFT_SHAPE[1]), dtype=np.float32)
                for i in range(mixed.shape[0]):
                    mixed_stft_np[i, :, :] = np.abs(librosa.core.stft(
                        np.squeeze(mixed_np[i, :]), n_fft=N_FFT, hop_length=HOP_LENGTH))

                noise_stft_np = np.empty(
                    (noise.shape[0], STFT_SHAPE[0], STFT_SHAPE[1]), dtype=np.float32)
                for i in range(noise.shape[0]):
                    noise_stft_np[i, :, :] = np.abs(librosa.core.stft(
                        np.squeeze(noise_np[i, :]), n_fft=N_FFT, hop_length=HOP_LENGTH))

                noise_stft = torch.from_numpy(noise_stft_np).to(DEV)

                # mixed_stft_db = librosa.amplitude_to_db(mixed_stft)

                if is_start:
                    print('\n(1)\t', time() - t)
                ####################################################
                if is_start:
                    t = time()

                noise_pre_stft = self.policy_net(
                    torch.from_numpy(mixed_stft_np).float().to(DEV))
                noise_pre_stft_np = noise_pre_stft.clone().detach().cpu().numpy()

                if is_start:
                    print('(2)\t', time() - t)
                ####################################################
                if is_start:
                    t = time()

                noise_pre_np = np.empty(
                    mixed.shape, dtype=np.float32)

                for i in range(mixed.shape[0]):
                    noise_pre_np[i, :] = librosa.griffinlim(
                        np.squeeze(noise_pre_stft_np[i, :, :]), hop_length=8)

                #     noise_pre = torch.from_numpy(librosa.griffinlim(
                #         noise_pre_stft[].detach().cpu().numpy(), hop_length=HOP_LENGTH))

                # noise_pre = F.griffinlim(noise_pre_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
                #                          n_iter=32, power=2, momentum=0.99, window=torch.ones(N_FFT).to(DEV), length=None, rand_init=True).to(DEV)
                # noise_pre = self.griffinlim(
                #     torch.squeeze(noise_pre_stft.clone().detach()[0, :, :])).to(DEV)

                # noise_pre_stft = self.policy_net(
                #     torch.from_numpy(mixed_stft).to(DEV))

                # noise_pre_stft_np = noise_pre_stft.detach().cpu().numpy()

                if is_start:
                    print('(3)\t', time() - t)
                ####################################################
                if is_start:
                    t = time()

                pesq_soc = 0.0
                # noise_pre = np.empty(
                #     vocal.shape, dtype=np.float64)
                # vocal_pre = np.empty(
                #     mixed.shape, dtype=np.float64)
                # noise_pre = torch.emtpy(
                #     (mixed.shape[0], SIGNAL_BOUND), dtype=mixed.dtype, device=DEV)
                # vocal_pre = torch.emtpy(
                #     (mixed.shape[0], SIGNAL_BOUND), dtype=mixed.dtype, device=DEV)

                # try:
                # **

                # except Exception as ex:
                #     print(type(ex), ex)
                #     print(torch.cuda.memory_summary())
                noise_pre = torch.from_numpy(noise_pre_np).to(DEV)

                vocal_pre = mixed - noise_pre
                vocal_pre_np = vocal_pre.detach().cpu().numpy()

                # try:
                for i in range(mixed.shape[0]):
                    # noise_pre_i = librosa.griffinlim(
                    #     np.squeeze(noise_pre_stft_np[i, :]), hop_length=HOP_LENGTH)
                    # vocal_pre_i = np.squeeze(mixed_np[i, :]) - noise_pre_i

                    pesq_soc = pesq_soc + pesq_func(rate_np, np.squeeze(vocal_np[i, :]),
                                                    np.squeeze(vocal_pre_np[i, :]))

                    # noise_pre[i, :] = noise_pre_i
                    # vocal_pre[i, :] = vocal_pre_i

                # except Exception as ex:
                #     print(type(ex), ex)
                #     continue

                pesq_soc = pesq_soc / mixed.shape[0]

                if is_start:
                    print('(4)\t', time() - t)
                    # print(vocal.shape, noise_pre.shape)
                    # print(noise.shape, vocal_pre.shape)
                    # print(type(vocal), type(vocal_pre))
                    # print(type(noise), type(noise_pre))

                ####################################################
                # * update policy
                if is_start:
                    t = time()

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

                # loss_v = self.loss_func(
                #     Variable(vocal), Variable(vocal_pre))
                # loss_n = self.loss_func(
                #     Variable(noise), Variable(noise_pre))
                loss = self.loss_func(
                    noise_stft, noise_pre_stft)

                # loss = loss_v + loss_n  # + loss_ns
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.pesq_lst.append(pesq_soc)
                self.loss_lst.append(loss.detach().cpu().numpy())

                if is_start:
                    print('(5)\t', time() - t)
                ####################################################
                is_start = 0

            show_plt('loss_' + str(epoch), self.loss_lst, self.saving_path)
            show_plt('pesq_' + str(epoch), self.pesq_lst, self.saving_path)
            save_flac(self.saving_path, str(epoch) +
                      '_vocal_pre.flac', np.squeeze(vocal_pre[0, :].detach().cpu().numpy()), rate_np)
            save_flac(self.saving_path, str(epoch) +
                      '_mixed_ori.flac', np.squeeze(mixed[0, :].detach().cpu().numpy()), rate_np)
            save_flac(self.saving_path, str(epoch) +
                      '_noise_pre.flac', np.squeeze(noise_pre[0, :].detach().cpu().numpy()), rate_np)
            # save_flac(saving_path, str(epoch) + '_noise.flac',
            #           np.real(np.fft.ifft(np.squeeze(noise_pre_f.detach().cpu().numpy()[0, :]))), rate)

            save_model(self.policy_net.state_dict(),
                       self.saving_path, 'model.pt')


# %%
if __name__ == '__main__':
    # with open("config.yaml") as fp:
    #     cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # print(cfg)

    dataset = AudioDataset(TRAIN_DATA_PATH, TRAIN_CLASS, SIGNAL_BOUND)
    train_num = floor(dataset.__len__() * TRAIN_PER)
    print('   device:\t', DEV)
    print('train_num:\t', train_num)

    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_num, dataset.__len__() - train_num])

    train_loader = DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

    # train_loader, val_loader = AudioDataset.get_dataloader(train_set, val_set)

    se = SpeechEnhancement(train_loader=train_loader, val_loader=val_loader)
    se.train()


# %%
