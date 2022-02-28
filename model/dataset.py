from turtle import back
from torch.utils.data import Dataset, DataLoader
import torch
# from torchaudio.backend.soundfile_backend import load
import os
# import soundfile as sf
import librosa
import numpy as np
from config import *


class AudioDataset(Dataset):

    def __init__(self, root, data_class, signal_bound):
        super(AudioDataset, self).__init__()
        # self.transform = transform
        self.signal_bound = signal_bound
        self.root = root
        # self.hop_length = hop_length
        # self.n_fft = n_fft
        self.vocal_pth = os.path.join(self.root, 'vocal')
        self.mixed_pth = os.path.join(self.root, 'mixed')
        self.data_class = data_class
        # self.train_data = sorted(glob.glob(os.path.join(root, "/*.*")))
        self.mixed_lst = os.listdir(os.path.join(self.mixed_pth, data_class))
        self.vocal_lst = os.listdir(os.path.join(self.vocal_pth, data_class))

        self.data_len = len(self.mixed_lst)

    def __getitem__(self, index):
        '''
        return mixed, vocal, noise, rate
        '''
        # os.path.join(self.root, self.list[index])
        m = torch.zeros((self.signal_bound,))
        v = torch.zeros((self.signal_bound,))

        # mixed, rate = load(os.path.join(
        #     self.mixed_pth, self.data_class, self.mixed_lst[index]))
        # vocal, rate = load(os.path.join(
        #     self.vocal_pth, self.data_class, self.vocal_lst[index]))

        # ? librosa
        mixed, rate = librosa.load(os.path.join(
            self.mixed_pth, self.data_class, self.mixed_lst[index]), sr=SAMPLE_RATE)

        vocal, rate = librosa.load(os.path.join(
            self.vocal_pth, self.data_class, self.vocal_lst[index]), sr=SAMPLE_RATE)

        if mixed.shape[0] > self.signal_bound:
            m = mixed[:self.signal_bound]
            v = vocal[:self.signal_bound]
        else:
            m = np.concatenate((mixed, mixed, mixed, mixed))[
                :self.signal_bound]
            v = np.concatenate((vocal, vocal, vocal, vocal))[
                :self.signal_bound]

        n = m-v

        # m_stft = np.abs(librosa.stft(m, n_fft=self.n_fft, hop_length=self.hop_length))
        # m_stft_db = librosa.amplitude_to_db(m_stft)
        # v_stft = np.abs(librosa.stft(v, n_fft=self.n_fft, hop_length=self.hop_length))
        # v_stft_db = librosa.amplitude_to_db(v_stft)
        # n_stft = np.abs(librosa.stft(n, n_fft=self.n_fft, hop_length=self.hop_length))
        # n_stft_db = librosa.amplitude_to_db(n_stft)

        # audio_stft_magnitude, audio_stft_phase = librosa.magphase(audio_stft)
        # audio_stft_magnitude_db = librosa.amplitude_to_db(audio_stft_magnitude)

        return m, v, n, rate

    def __len__(self):
        return self.data_len

    # def get_dataloader(self, train_set, val_set, is_shuffle=True):
    #     train_loader = DataLoader(
    #         dataset=train_set, batch_size=BATCH_SIZE, shuffle=is_shuffle)
    #     val_loader = DataLoader(
    #         dataset=val_set, batch_size=BATCH_SIZE, shuffle=is_shuffle)

    #     return train_loader, val_loader
