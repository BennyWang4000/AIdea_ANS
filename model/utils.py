from torch import save
from pesq import pesq
from matplotlib import pyplot as plt
import soundfile as sf
import os
import librosa
import numpy as np
from config import *

'''
pesq: -0.5 to 4.5
# reward_pesq: (1 to 6)^ 2    = 1 to 36
# if batch= 4: (1 to 6)^ 2* 4 = 4 to 144
'''


def mul_dim_stft(audio_np, batch_size):
    stft_np = np.empty(
        (batch_size, STFT_SHAPE[0], STFT_SHAPE[1]), dtype=np.float32)
    for i in range(batch_size):
        stft_np[i, :, :] = np.abs(librosa.core.stft(
            np.squeeze(audio_np[i, :]), n_fft=N_FFT, hop_length=HOP_LENGTH))

    # mixed_stft_db = librosa.amplitude_to_db(mixed_stft)
    return stft_np


def mul_dim_griffinlim(stft_np, batch_size):
    istft_np = np.empty(
        (batch_size, SIGNAL_BOUND), dtype=np.float32)

    for i in range(batch_size):
        istft_np[i, :] = librosa.griffinlim(
            np.squeeze(stft_np[i, :, :]), hop_length=HOP_LENGTH)

    return istft_np


def reward_func(pesq_soc):
    '''
    return 0.1 to 0
    '''
    return (0.02 * (6.0 / (pesq_soc + 1.5))) - 0.02


# def reward_func(rate, ori, denoise):
#     '''
#     return 0.1 to 0
#     '''
#     return (0.02 * (6.0 / (pesq(rate, ori, denoise, 'wb') + 1.5))) - 0.02


def pesq_func(rate, ori, denoise):
    return pesq(rate, ori, denoise, 'wb')


def fourier_bound(data, bound):
    return data[:bound]


def get_saving_path(path):
    num = 1
    while os.path.exists(os.path.join(path, 'exp' + str(num))):
        num = num + 1

    os.mkdir(os.path.join(path, 'exp' + str(num)))

    return os.path.join(path, 'exp' + str(num))


def show_plt(name, data, path):
    plt.figure()
    plt.plot(data)
    plt.title(name)
    plt.show
    plt.savefig(os.path.join(path, name + '.jpg'))


def show_plt_via_lst(name_lst, data_lst, path):
    for i in range(len(name_lst)):
        show_plt(name_lst[i], data_lst[i], path)


def save_flac(name, data, path, rate):
    sf.write(os.path.join(path, name), data, rate, format='FLAC')


def save_flac_via_lst(name_lst, data_lst, path, rate):
    for i in range(len(name_lst)):
        save_flac(name_lst[i], data_lst[i], path, rate)


def save_model(state, path, name):
    save(state, os.path.join(path, name))


# def get_spectrograms(audio, rate, n_fft=255, hop_length=8048, n_mels=512):
#     '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
#     Args:
#       sound_file: A string. The full path of a sound file.

#     Returns:
#       mel: A 2d array of shape (T, n_mels) <- Transposed
#       mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
#  '''
#     # stft
#     linear = librosa.stft(y=audio,
#                           n_fft=n_fft)
#     # linear = librosa.stft(y=audio,
#     #                       n_fft=n_fft,
#     #                       hop_length=hop_length,
#     #                       win_length=win_length)

#     # magnitude spectrogram
#     mag = np.abs(linear)  # (1+n_fft//2, T)

#     # mel spectrogram
#     mel_basis = librosa.filters.mel(
#         rate, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
#     mel = np.dot(mel_basis, mag)  # (n_mels, t)

#     # to decibel
#     mel = 20 * np.log10(np.maximum(1e-5, mel))
#     mag = 20 * np.log10(np.maximum(1e-5, mag))

#     # # normalize
#     # mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
#     # mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

#     # # Transpose
#     # mel = mel.T.astype(np.float32)  # (T, n_mels)
#     # mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

#     return mel, mag


# # ====================================================
# def audio_to_magnitude_db_and_phase(audio, n_fft=255, hop_length_fft=8064):
#     """This function takes an audio and convert into spectrogram,
#        it returns the magnitude in dB and the phase"""

#     stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
#     stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

#     stftaudio_magnitude_db = librosa.amplitude_to_db(
#         stftaudio_magnitude, ref=np.max)

#     return stftaudio_magnitude_db, stftaudio_phase

# def magjitude_db_to_audio_via_grillim(stftaudio_magnitude_db):
#     pass

# def magnitude_db_and_phase_to_audio(stftaudio_magnitude_db, stftaudio_phase, frame_length=8064, hop_length_fft=8064):
#     """This functions reverts a spectrogram to an audio"""

#     stftaudio_magnitude_rev = librosa.db_to_amplitude(
#         stftaudio_magnitude_db, ref=1.0)

#     # taking magnitude and phase of audio
#     audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
#     # audio_reverse_stft = stftaudio_magnitude_rev * np.exp(stftaudio_phase * 1j)
#     audio_reconstruct = librosa.core.istft(
#         audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

#     return audio_reconstruct


# def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
#     """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
#     a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
#     (nb_frame,dim_square_spec,dim_square_spec)"""

#     nb_audio = numpy_audio.shape[0]

#     m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
#     m_phase = np.zeros(
#         (nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

#     for i in range(nb_audio):
#         m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
#             n_fft, hop_length_fft, numpy_audio[i])

#     return m_mag_db, m_phase
