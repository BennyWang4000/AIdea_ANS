# %%
import glob
from config import *
from statistics import mode
import soundfile as sf
from pesq import pesq
import os
import yaml
import torch
from model.dataset import AudioDataset
from matplotlib import pyplot as plt
import tqdm
import scipy.fft
import model.utils as utils
from math import floor
import numpy as np
# import torchvision.transforms as trns
import torchaudio.transforms as T
from scipy import signal
import scipy
import librosa
import librosa.display
import torchaudio

# import scipy.io.wavfile

from model.models import UNet, WaveGANGenerator
from torchsummary import summary

# with open("config.yaml") as fp:
#     cfg = yaml.load(fp, Loader=yaml.FullLoader)

TRAIN_DATA_PATH = 'D://CodeRepositories//py_project//aidea//ANS//data//train'
SAVING_PATH = 'C://Users//costco//Desktop'
SAMPLE_NAME = 'mixed_01069_dog_bark.flac'
SAMPLE_NAME_2 = 'vocal_01069.flac'
print('done')
#%%
tup= (0 , 1, 2, 3)
print(tup.shape[0])
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
# ? model
# enc= Encoder()
# dec= Decoder()
# print(enc)
# print('===========')
# print(dec)
model = UNet()

print(summary(model, (129, 12376), batch_size=4))
# %%
model = WaveGANGenerator(
    slice_len=32768,
    model_size=32,
    use_batch_norm=True,
    num_channels=1,
)
# print(model)

print(summary(model, (8, 32768), batch_size=4))
# print(model)
# %%
# ? audio signal
data, rate = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
data2, rate2 = sf.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))
# rate, data  = scipy.io.wavfile.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
# rate2, data2  = scipy.io.wavfile.read(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))
# sf.write(os.path.join(SAVING_PATH, 'zzori.flac'),
#          data, rate, format='FLAC')

plt.figure()
plt.plot(data)

plt.figure()
plt.plot(data2)
# %%


#? torchaudio
spectrogram = T.Spectrogram(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    power=2.0,
)
# Perform transformation
# spec = spectrogram(waveform)


#%%
# ? librosa
l_data, _ = librosa.load(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))

print(l_data.shape, l_data.dtype, type(l_data))
# l_f = librosa.stft(l_data, n_fft=255)
magnitude_db, phase = utils.audio_to_magnitude_db_and_phase(
    l_data, n_fft=256, hop_length_fft=64)
print(magnitude_db.shape, magnitude_db.dtype, type(magnitude_db))


# l_f_db= librosa.core.amplitude_to_db(l_f, ref=1.0, amin=1e-20, top_db=80.0)
# ? phase
audio_reconstruct = utils.magnitude_db_and_phase_to_audio(
    magnitude_db, phase, frame_length=64, hop_length_fft=64)
print(audio_reconstruct.shape, audio_reconstruct.dtype, type(audio_reconstruct))

# plt.figure()
# plt.plot(magnitude_db)
# plt.figure()
# plt.plot(phase)
plt.figure()
plt.title('audio_reconstruct')
plt.plot(audio_reconstruct)
# plt.pcolormesh(audio_reconstruct)
plt.figure()
plt.title('ori_data')
plt.plot(l_data)


# %%
# ? griffin lim ******************************
l_data, rate = librosa.load(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
t_data= torch.from_numpy(l_data)
# t_data, rate= torch.backend
# v_data, rate = librosa.load(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME_2))
print(l_data.shape)
print(t_data.shape)
# n_data = l_data - v_data
# l_data = np.concatenate((l_data, l_data))

audio_stft = np.abs(librosa.stft(l_data, n_fft=N_FFT, hop_length=HOP_LENGTH))
torch_stft = torch.abs(torch.stft(t_data, n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=False).sum(-1))
plt.figure()
plt.title('audio_stft')
plt.plot(audio_stft)
plt.figure()
plt.title('torch_stft')
plt.plot(torch_stft)
print(audio_stft.dtype, type(audio_stft), audio_stft.shape)
print(torch_stft.dtype, type(torch_stft), torch_stft.shape)
# audio_stft_db = librosa.amplitude_to_db(audio_stft)
# print(audio_stft_db.dtype, type(audio_stft_db))
# v_stft = np.abs(librosa.stft(v_data, n_fft=255, hop_length=8))
# n_stft = audio_stft - v_stft
# v_stft_db = librosa.amplitude_to_db(n_stft)


# audio_stft_magnitude, audio_stft_phase = librosa.magphase(audio_stft)
# audio_stft_magnitude_db = librosa.amplitude_to_db(audio_stft_magnitude)

trans= torchaudio.transforms.GriffinLim(
            n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEV)

# griffin
# leng = l_data.shape[0]
audio_inv = librosa.griffinlim(audio_stft, hop_length=8)
torch_inv= trans(torch_stft)

print(audio_inv.dtype, type(audio_inv), audio_inv.shape)
print(torch_inv.dtype, type(torch_inv), torch_inv.shape)

plt.figure()
plt.title('audio_inv')
plt.plot(audio_inv)
plt.figure()
plt.title('torch_inv')
plt.plot(torch_inv)

# audio_inv_db = librosa.griffinlim(audio_stft_db, hop_length=8)
# print(audio_inv_db.dtype, type(audio_inv_db), audio_inv_db.shape)
# n_inv_db = librosa.griffinlim(n, hop_length=64)
# audio_inv_mag_db = librosa.griffinlim(audio_stft_magnitude_db, hop_length=64)


# audio_stft_magnitude_rev = librosa.db_to_amplitude(audio_stft_magnitude_db)
# audio_reverse_stft = audio_stft_magnitude_rev * audio_stft_phase
# audio_reverse_stft_1j = audio_stft_magnitude_rev * \
#     np.exp(audio_stft_phase * 1j)
# # istft
# audio_istft = librosa.istft(audio_stft, hop_length=64)
# audio_istft_db = librosa.istft(audio_stft_db, hop_length=64)
# audio_istft_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=64)
# audio_istft_reconstruct_1j = librosa.core.istft(
#     audio_reverse_stft_1j, hop_length=64)

# sf.write(os.path.join(SAVING_PATH, 'n.flac'),
#          n_inv_db, rate, format='FLAC')

sf.write(os.path.join(SAVING_PATH, 'audio_inv.flac'),
         audio_inv, rate, format='FLAC')
sf.write(os.path.join(SAVING_PATH, 'torch_inv.flac'),
         torch_inv, rate, format='FLAC')
# sf.write(os.path.join(SAVING_PATH, 'audio_inv_db.flac'),
#          audio_inv_db, rate, format='FLAC')
# sf.write(os.path.join(SAVING_PATH, 'audio_inv_mag_db.flac'),
#          audio_inv_mag_db, rate, format='FLAC')

# sf.write(os.path.join(SAVING_PATH, 'audio_istft.flac'),
#          audio_istft, rate, format='FLAC')
# sf.write(os.path.join(SAVING_PATH, 'audio_istft_db.flac'),
#          audio_istft_db, rate, format='FLAC')
# sf.write(os.path.join(SAVING_PATH, 'audio_istft_reconstruct.flac'),
#          audio_istft_reconstruct, rate, format='FLAC')
# sf.write(os.path.join(SAVING_PATH, 'audio_istft_reconstruct_1j.flac'),
#          audio_istft_reconstruct_1j, rate, format='FLAC')
# plt.figure()
# plt.title('ori_data')
# plt.plot(l_data)
# plt.figure()
# plt.title('audio_stft')
# plt.plot(audio_stft)
# plt.figure()
# plt.title('audio_stft_db')
# plt.plot(audio_stft_db)

# plt.figure()
# plt.title('audio_istft')
# plt.plot(audio_istft)

# %%
# ? torch stft
sig, rate = librosa.load(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME))
sig = np.concatenate((sig, sig, sig, sig))[:99000]
sig_t = torch.from_numpy(sig)

sig_tt = torch.stft(sig_t, n_fft=256, hop_length=8)

# %%
# ? mel to stft

sig, rate = librosa.load(os.path.join(TRAIN_DATA_PATH, SAMPLE_NAME), sr=16000)
sig = np.concatenate((sig, sig, sig, sig))[:99000]


print(sig.shape)
# plt.figure()
# plt.title('ori')
# plt.plot(sig)

window = 'han'
hop_length = 8
n_fft = 256
# mel = librosa.feature.melspectrogram(
#     sig, sr=rate, n_fft=n_fft, hop_length=hop_length)

st = np.abs(librosa.stft(
    sig, n_fft=n_fft, hop_length=hop_length))

# print(mel.shape, type(mel), mel.dtype)
print(st.shape, type(st), st.dtype)

leng = sig.shape[0]
# mel_inv = librosa.griffinlim(mel, hop_length=hop_length, length=leng)
st_inv = librosa.griffinlim(st, hop_length=hop_length, length=leng)

# sf.write(os.path.join(SAVING_PATH, 'mel_32.flac'),
#          mel_inv, rate, format='FLAC')

sf.write(os.path.join(SAVING_PATH, 'st_16.flac'),
         st_inv, rate, format='FLAC')

# plt.figure()
# plt.title('melspectrogram')
# plt.plot(spec)
# ************
# * plt.figure()
# * a = librosa.display.specshow(spec, sr=16000, fmax=8000)
# * plt.colorbar()
# * print(type(a))
# * spec = librosa.griffinlim(spec, hop_length=64)
# * plt.figure()
# * plt.title('griffinlim')
# * plt.plot(spec)
# ************
# spec = librosa.feature.inverse.mel_to_stft(spec)
# plt.figure()
# plt.title('mel_to_stft')
# plt.plot(spec)
# %%
# ? spectrogram
'''
f.shape= half of nperseg+ 1
'''
f, t, z = signal.stft(data, fs=64000, nperseg=256)
print(f.shape, t.shape, z.shape)
# # t, zi = signal.istft(np.real(z), fs=1024)

# f2, t2, data2_sp= signal.spectrogram(z, 1024)
# print(f2.shape, t2.shape)
# print(data2_sp.shape, data2_sp.dtype)

# * Plot
plt.figure()
plt.pcolormesh(t, f, np.real(z) ** 2)
plt.figure()
plt.plot(z)

# print(p.shape)
# plt.figure()
# p, f, b, i = plt.specgram(data2, Fs=1024)
# print(p.shape, f.shape, b.shape, type(i))

# plt.figure()
# plt.plot(data2_sp)
# %%
# ? spectrogram
# print(data2.shape)
# audio = scipy.mean(data2, axis=0)
f, t, X = signal.spectrogram(data2, fs=1000, nperseg=256, mode='magnitude')

print(X.shape)
plt.figure()
plt.plot(X)
plt.figure()
plt.pcolormesh(t, f, 10*np.log10(X[:]))
plt.figure()
plt.figure()
plt.specgram(data2)
# fig = pylab.figure()
# ax = pylab.Axes(fig, [0,0,1,1])
# ax.set_axis_off()
# fig.add_axes(ax)
# pylab.imshow(scipy.absolute(X.T), origin='lower', aspect='auto', interpolation='nearest')

# %%
# ? minus
sf.write(os.path.join(SAVING_PATH, 'no.flac'),
         data - data2, rate, format='FLAC')


# %%
# ?stft
# f, t, z = signal.stft(data, 12000)
# plt.figure(5)
# plt.plot(z)
# # print()
# # plt.pcolormesh(t, f, np.abs(z))
# # print(f.shape, type(f), f.dtype)
# # print(t.shape, type(t), t.dtype)
# print(z.shape, type(z), z.dtype)
print(data.shape, data2.shape)
f, t, z = signal.stft(data2, fs=44000, nperseg=62)
f, t, z1 = signal.stft(data, fs=44000, nperseg=62)
rate = 12000
plt.figure(6)
plt.plot(z)
print(f.shape, type(f), f.dtype)
print(t.shape, type(t), t.dtype)
print(z.shape, type(z), z.dtype)
print(z1.shape, type(z1), z1.dtype)

t, zi = signal.istft(np.real(z), fs=44000, nperseg=62)
plt.figure(8)
plt.plot(zi)
plt.figure()
plt.plot(data2)
print(t.shape, type(t), t.dtype)
print(zi.shape, type(zi), zi.dtype)
print(data2.shape)


sf.write(os.path.join(SAVING_PATH, 'ori.flac'),
         data2, rate, format='FLAC')

sf.write(os.path.join(SAVING_PATH, 'isf.flac'),
         zi, rate, format='FLAC')

# %%
# %%
# ? padding

zz = np.zeros((100000,))
zz[:data.shape[0]] = data

zz[data.shape[0]:] = data[:100000-data.shape[0]]
print(data.shape, data2.shape)
print(rate, rate2)
print(zz[0], zz[1], zz[data.shape[0]], zz[data.shape[0] + 1])
plt.figure(1)
# plt.specgram(data)
# sf.write(os.path.join(SAVING_PATH, 'zz.flac'),
#          zz, rate, format='FLAC')
print(zz.shape, ', ', data.shape)

plt.figure(2)
plt.plot(data)
plt.plot(data2)
# print(pesq(rate, data, data2, 'wb'))
# print(pesq(rate, zz, data2, 'wb'))

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
# # %%
# transform = trns.Compose(
#     trns.ToTensor
# )

# print(cfg.get('train_data_path'))
# train_dataset = AudioDataset(transform, cfg['train_data_path'],
#                              cfg['class'], cfg['signal_bound'])

# dataloader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=cfg.get('batch_size'), shuffle=True)

# for epoch in range(cfg.get('epochs')):
#     # progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
#     for i, data in enumerate(dataloader):
#         print(data)
#         print(type(data))

#         break
# print('f')
# # %%

# dataset = AudioDataset(transform, cfg['train_data_path'],
#                        cfg['class'], cfg['signal_bound'])

# train_num = floor(dataset.__len__() * cfg['train_per'])
# print('train_num:\t', train_num)
# train_set, val_set = torch.utils.data.random_split(
#     dataset, [train_num, dataset.__len__() - train_num])

# dataloader = torch.utils.data.DataLoader(
#     dataset=train_set, batch_size=cfg.get('batch_size'), shuffle=True)


# # %%
# root = cfg.get('train_data_path')
# ist = os.listdir(root)

# print(ist)

# # %%
