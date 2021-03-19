import matplotlib.pyplot as plt
import csv
import pandas as pd
import pickle
import math
import os
import numpy as np
import os.path
from util import *
# Code to get relevant arrays
def calculate_batch_sample(epoch, batch_per_epoch, sample_interval):
    """
    The audio files are sampled not after some number of epochs, but after some
    number of batches. One epoch (when training on the full dataset for example),
    are 688 batches. This function return the batch number that correspond to a
    particular epoch the audio samples of which we want to investigate.
    """
    epoch = epoch + 1
    upper = (batch_per_epoch * epoch )// sample_interval
    lower = math.ceil((batch_per_epoch * (epoch-1) )/ sample_interval)
    output = []
    for factor in range(lower, upper+1):
        output.append(factor*sample_interval)
    return output
def get_relevant_arrs(epochs, path, batch_per_epoch, sample_interval):
    batches = []
    for epoch in epochs:
        batches += [calculate_batch_sample(epoch, batch_per_epoch, sample_interval)[0]]
    for batch in batches:
        def save_arr(x,test_train='test'):
            save_path = path + f'{test_train}/{batch}/'
            mkdir(save_path)

            def save_generated_and_source(sample, instrumentA, instrumentB, main_file_name='pp'):
                nonlocal test_train, save_path
                # save the generatetd audio array
                arr = sample['arr']
                np.save(save_path+f'{main_file_name}.npy', arr)

                # save the source audio arrays
                A_name = sample['A'] # 'piano_piano_0.wav_5'
                B_name = sample['B']
                if test_train == 'test':
                    path_to_source_audio = '../arbitrary_timbre_transfer/data/spec2test/'
                elif test_train == 'train':
                    path_to_source_audio = '../arbitrary_timbre_transfer/data/spec2/train/'
                mag_A = np.load(path_to_source_audio+f'{instrumentA}/mag/'+A_name+'.npy')
                phase_A = np.load(path_to_source_audio+f'{instrumentA}/phase/'+A_name+'.npy')
                arr_A = build_audio(mag_A, phase_A)
                mag_B = np.load(path_to_source_audio+f'{instrumentB}/mag/'+B_name+'.npy')
                phase_B = np.load(path_to_source_audio+f'{instrumentB}/phase/'+B_name+'.npy')
                arr_B = build_audio(mag_B, phase_B)
                np.save(save_path+f'{main_file_name}A.npy', arr_A)
                np.save(save_path+f'{main_file_name}B.npy', arr_B)

            # piano - piano
            appears = False
            for sample in x:
                if sample['homo'] and 'piano' in sample['A']:
                    pp_arr = sample['arr']
                    appears = True
                    break
            if appears:
                save_generated_and_source(sample, instrumentA='piano', instrumentB='piano', main_file_name='pp')

            # flute - flute
            appears = False
            for sample in x:
                if sample['homo'] and 'flute' in sample['A']:
                    ff_arr = sample['arr']
                    appears = True
                    break
            if appears :
                save_generated_and_source(sample, instrumentA='flute', instrumentB='flute', main_file_name='ff')

            # piano - flute
            appears = False
            for sample in x:
                if (not sample['homo']) and 'piano' in sample['A']:
                    pf_arr = sample['arr']
                    appears = True
                    break
            if appears:
                save_generated_and_source(sample, instrumentA='piano', instrumentB='flute', main_file_name='pf')

            # flute - piano
            appears = False
            for sample in x:
                if (not sample['homo']) and 'flute' in sample['A']:
                    fp_arr = sample['arr']
                    appears = True
                    break
            if appears:
                save_generated_and_source(sample, instrumentA='flute', instrumentB='piano', main_file_name='fp')

        with open(path+f'kaggle_out/test_{batch}.pkl', 'rb') as f:
            x = pickle.load(f)

        save_arr(x, test_train='test')

        if os.path.exists(path+f'kaggle_out/train_{batch}.pkl'):
            with open(path+f'kaggle_out/train_{batch}.pkl', 'rb') as f:
                x = pickle.load(f)

            save_arr(x, test_train='train')

def get_relevant_data_for_exp(path_for_exp, high_loss_epoch_num, low_loss_epoch_num):
    high_loss_epoch = [high_loss_epoch_num]
    low_loss_epoch = [ low_loss_epoch_num]
    epoch_0 = [0]

    get_relevant_arrs(epoch_0, path = path_for_exp,batch_per_epoch=688, sample_interval=600)

    get_relevant_arrs(high_loss_epoch, path = path_for_exp,batch_per_epoch=688, sample_interval=600)

    get_relevant_arrs(low_loss_epoch, path = path_for_exp,batch_per_epoch=688, sample_interval=600)
get_relevant_data_for_exp('./fulldat/audio_samples/exp1/', 26, 20)
get_relevant_data_for_exp('./fulldat/audio_samples/exp2/', 29, 14)
get_relevant_data_for_exp('./fulldat/audio_samples/exp3/', 9, 26)
get_relevant_data_for_exp('./fulldat/audio_samples/exp4/', 27, 19)
get_relevant_data_for_exp('./fulldat/audio_samples/exp5/', 18, 17)
get_relevant_data_for_exp('./fulldat/audio_samples/exp6/', 15, 17)
get_relevant_data_for_exp('./fulldat/audio_samples/exp7/', 15, 23)
get_relevant_data_for_exp('./fulldat/audio_samples/exp8/', 25, 27)
