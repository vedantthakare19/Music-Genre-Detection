# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:01:28 2019

@author: SEVVAL
"""
 #    *** DATA PROCESSING            
 
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
import warnings


#feature names oluşturuldu
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1,21):  #21 yap
    header += f' mfcc{i}'
header += ' label'
header = header.split()
#%% audio dosyalar dan librosa kütüphanesi ile feature çıkarımı yapıldı
file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'./{g}'):
        songname = f'./{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())        
#%%  Visualizing Audio

import librosa
ses = "blues.00000.wav"
y,sr = librosa.load(ses)
print("y type:",type(y),"sr type",type(sr))
print(y.shape,sr)
import IPython.display as ipd
ipd.Audio(ses)
ipd.Audio(y,rate=sr)
#%%  Playing Audio
import IPython.display as ipd
ipd.Audio(ses)
ipd.Audio(y,rate=sr)
#%%  Visualizing Audio
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14,3))
librosa.display.waveplot(y,sr=sr)
#%%  Spectrogram

X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 3))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

#%%  Zero Crossing Rate
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 3))
plt.plot(y[n0:n1])
plt.grid()
zero_crossings = librosa.zero_crossings(y[n0:n1], pad=False)
print(sum(zero_crossings))    #metal8   blues16

#%% Spectral Centroid
import  matplotlib.pyplot as plt, IPython.display as ipd, sklearn
spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation

def normalize(y, axis=0):

    return sklearn.preprocessing.minmax_scale(y, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
#%% Spectral Rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]
librosa.display.waveplot(y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='y')
#%% MFCC
x, fs = librosa.load(ses)
librosa.display.waveplot(x, sr=sr)
mfccs = librosa.feature.mfcc(y, sr=fs)

#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
#%% chroma frequencies
hop_length = 512
chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=hop_length)
plt.figure(figsize=(10, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
#%%  şarkıyı çaldırma
import simpleaudio as sa

wave_obj = sa.WaveObject.from_wave_file("classical.00000.wav")
play_obj = wave_obj.play()
play_obj.wait_done()


#%%      oluşturulan csv dosyasını okuma
   
import pandas as pd
data = pd.read_csv("data.csv")