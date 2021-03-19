import matplotlib.pyplot as plt
import csv
import pandas as pd
import pickle
import math
import IPython.display as ipd
import IPython
import os
import numpy as np
import os.path
import librosa


# credits: https://github.com/hmartelb/Pix2Pix-Timbre-Transfer

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# reconstruct audio from np arrays
def build_audio(mag, phase, n_fft=1024):
    y = np.expand_dims(mag, axis=0)
    y = np.expand_dims(y, axis=-1)

    prediction = (y + 1) / 2

    mag_db = join_magnitude_slices(prediction, phase.shape)
    mag = db_to_amplitude(mag_db)
    audio_out = inverse_transform(mag, phase, n_fft)
    return audio_out

def join_magnitude_slices(mag_sliced, target_shape):
    mag = np.zeros((mag_sliced.shape[1], mag_sliced.shape[0]*mag_sliced.shape[2]))
    for i in range(mag_sliced.shape[0]):
        mag[:,(i)*mag_sliced.shape[2]:(i+1)*mag_sliced.shape[2]] = mag_sliced[i,:,:,0]
    mag = mag[0:target_shape[0], 0:target_shape[1]]
    return mag

def db_to_amplitude(mag_db, amin=1/(2**16), normalize=True):
    if(normalize):
        mag_db *= 20*np.log1p(1/amin)
    return amin*np.expm1(mag_db/20)


def inverse_transform(mag, phase, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    if(normalize):
        mag = mag * np.sum(np.hanning(nfft)) / 2
    if(crop_hf):
        mag = add_hf(mag, target_shape=(phase.shape[0], mag.shape[1]))
    R = mag * np.exp(1j*phase)
    audio = librosa.istft(R, hop_length=int(nfft/2), window=window)
    return audio



def add_hf(mag, target_shape):
    rec = np.zeros(target_shape)
    rec[0:mag.shape[0],0:mag.shape[1]] = mag
    return rec

def play_sample(arr):
    sr = 22050
    IPython.display.display(ipd.Audio(arr, rate=sr))
