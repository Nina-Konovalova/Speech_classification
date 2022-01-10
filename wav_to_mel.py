# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:48:12 2021

@author: nina_
"""

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


class Wav_to_mel():
    def __init__(self, path_to_wav, window_time=22, sample_rate=22050):

        self.sample_rate = sample_rate
        self.window_time = window_time
        self.w_l = int(self.sample_rate * 22 / 1000)

        self.path_to_wav = path_to_wav
        #self.path_to_npy = path_to_npy

    def augmentation(self, array):

        def speeding(array):
            speed_factor = np.random.randint(1, 3)
            return librosa.effects.time_stretch(array, speed_factor)

        def noisy(array, noise_factor=0.01):
            noise = np.random.randn(len(array))
            noise_factor = np.max(abs(array)) / 200
            augmented_data = array + noise_factor * noise
            # Cast back to same data type
            augmented_data = augmented_data.astype(type(array[0]))
            return augmented_data

        array = noisy(array)


        return array

    def scaled_array(self, array):
        minimum = np.min(array)
        maximum = np.max(array)
        return minimum, maximum, (array - minimum) / (maximum - minimum)

    def from_wav_to_mel(self, path, augmentation=True, scaling=True):

        amp, rate = librosa.load(path)
        plt.plot(amp)

        # augmantation
        if augmentation:
            amp = self.augmentation(amp)

        # create melspectrogram
        ref = librosa.feature.melspectrogram(amp, sr=self.sample_rate, window='hann', n_mels=128, win_length=self.w_l,
                                             hop_length=self.w_l // 4, fmin=1, fmax=8192)
        # scaling
        if (scaling):
            self.minimum, self.maximum, ref = self.scaled_array(ref)
        # transform to db
        audio_db = librosa.power_to_db(ref)

        return audio_db

    # Padding melspectrograms to the same length
    def pad_melspectrogram(self, audio_db):
        # print( self.length_to_padd)
        pad_length = self.length_to_padd - audio_db.shape[1]
        if pad_length > 0:
            audio_db_pad = np.pad(audio_db, (0, pad_length), mode='reflect')[:-pad_length, :]
        else:
            audio_db_pad = audio_db[:, :self.length_to_padd]
        return audio_db_pad

    def save_melspectrogram(self, audio_db, path):

        path = path[:-4] + '.npy'
        np.save(path, audio_db)

    def final_mels(self, scaling=True, padding=True, augmentation=True, length_to_padd=6223):
        path1 = self.path_to_wav
        audio_db = self.from_wav_to_mel(path1, augmentation=augmentation, scaling=scaling)

        if padding:
            self.length_to_padd = length_to_padd
            audio_db = self.pad_melspectrogram(audio_db)

        return audio_db

        #self.save_melspectrogram(audio_db, path)

    def load_melspectrogram(self, path, showplot=True):
        audio_db = np.load(path)
        if (showplot):
            plt.imshow(audio_db)
            plt.show()
        audio_db = librosa.db_to_power(audio_db)

        return audio_db

    def from_mel_to_wav(self, audio_db, path='listen.wav', showplot=True):

        y = librosa.feature.inverse.mel_to_audio(audio_db, sr=self.sample_rate, window='hann', win_length=self.w_l,
                                                 hop_length=self.w_l // 4, fmin=1, fmax=8192)
        if (showplot):
            plt.plot(y)
            plt.show()
        sf.write(path, y, self.sample_rate)
        return y

    def backward(self, path):
        audio_db = self.load_melspectrogram("/content/drive/MyDrive/Skoltech/Nanosemantica/Common voice/new_pad.npy")
        self.from_mel_to_wav(audio_db)


