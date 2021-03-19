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
from util import *
from data_generate import *
# Code to display
def display_audio_samples(epochs, path, batch_per_epoch, sample_interval):
    def display_samples(path_to_display):
        if os.path.exists(path_to_display+f'ppA.npy'):
            print('Piano - Piano')
            play_sample(np.load(path_to_display+f'ppA.npy'))
            play_sample(np.load(path_to_display+f'ppB.npy'))
            play_sample(np.load(path_to_display+f'pp.npy'))
        if os.path.exists(path_to_display+f'ffA.npy'):
            print('Flute - Flute')
            play_sample(np.load(path_to_display+f'ffA.npy'))
            play_sample(np.load(path_to_display+f'ffB.npy'))
            play_sample(np.load(path_to_display+f'ff.npy'))
        if os.path.exists(path_to_display+f'pfA.npy'):
            print('Piano - Flute')
            play_sample(np.load(path_to_display+f'pfA.npy'))
            play_sample(np.load(path_to_display+f'pfB.npy'))
            play_sample(np.load(path_to_display+f'pf.npy'))
        if os.path.exists(path_to_display+f'fpA.npy'):
            print('Flute - Piano')
            play_sample(np.load(path_to_display+f'fpA.npy'))
            play_sample(np.load(path_to_display+f'fpB.npy'))
            play_sample(np.load(path_to_display+f'fp.npy'))

    batches = []
    for epoch in epochs:
        batches += calculate_batch_sample(epoch, batch_per_epoch, sample_interval)
    for batch in batches:
        print('Playing test samples')
        display_samples(path_to_display=path+f'test/{batch}/')


        if os.path.exists(path+f'train/{batch}/'):
            print('Playing train samples')
            display_samples(path_to_display=path+f'train/{batch}/')



def show_exp(path_for_exp, high_loss_epoch_num, low_loss_epoch_num):
    high_loss_epoch = [high_loss_epoch_num]
    low_loss_epoch = [low_loss_epoch_num]
    epoch_0 = [0]

    print('Samples after epoch 0')
    display_audio_samples(epoch_0, path=path_for_exp, batch_per_epoch=688, sample_interval=600)
    print('*'*40)
    print('Samples of epoch with high G loss')
    display_audio_samples(high_loss_epoch, path=path_for_exp, batch_per_epoch=688, sample_interval=600)
    print('*'*40)
    print('Samples of epoch with low G loss')
    display_audio_samples(low_loss_epoch, path=path_for_exp, batch_per_epoch=688, sample_interval=600)

def play_4dat_audio(path_to_pickle_file):
    with open(path_to_pickle_file, 'rb') as f:
        x = pickle.load(f)

    for sample in x:
        A_name = sample['A'] # 'piano_piano_0.wav_5'
        B_name = sample['B']
        instrumentA = 'piano' if 'piano' in A_name else 'flute'
        instrumentB = 'piano' if 'piano' in B_name else 'flute'

        path_to_source_audio = './small_data_set/'

        mag_A = np.load(path_to_source_audio+f'{instrumentA}/mag/'+A_name+'.npy')
        phase_A = np.load(path_to_source_audio+f'{instrumentA}/phase/'+A_name+'.npy')
        arr_A = build_audio(mag_A, phase_A)
        mag_B = np.load(path_to_source_audio+f'{instrumentB}/mag/'+B_name+'.npy')
        phase_B = np.load(path_to_source_audio+f'{instrumentB}/phase/'+B_name+'.npy')
        arr_B = build_audio(mag_B, phase_B)

        print('*'*20)
        print(f'A: {instrumentA}, B: {instrumentB}')

        play_sample(arr_A)
        play_sample(arr_B)
        play_sample(sample['arr'])
