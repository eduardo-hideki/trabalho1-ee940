# EE940 - Trabalho 1
# Analise de sinais acusticos
# Eduardo Hideki Kobaicy e Gabriel Souza Murizine
# Codigo adaptado do 04-mirstat-mean.py

import pandas
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import librosa


x01, fs = librosa.load('./Martelo/M01.wav', sr=44100)
x02, fs = librosa.load('./Martelo/M02.wav', sr=44100)
x03, fs = librosa.load('./Martelo/M03.wav', sr=44100)
x04, fs = librosa.load('./Martelo/M04.wav', sr=44100)
x05, fs = librosa.load('./Martelo/M05.wav', sr=44100)
x06, fs = librosa.load('./Martelo/M06.wav', sr=44100)
x07, fs = librosa.load('./Martelo/M07.wav', sr=44100)
x08, fs = librosa.load('./Martelo/M08.wav', sr=44100)
x09, fs = librosa.load('./Martelo/M09.wav', sr=44100)
x10, fs = librosa.load('./Martelo/M10.wav', sr=44100)

y01, fs = librosa.load('./Passaro/P01.wav', sr=44100)
y02, fs = librosa.load('./Passaro/P02.wav', sr=44100)
y03, fs = librosa.load('./Passaro/P03.wav', sr=44100)
y04, fs = librosa.load('./Passaro/P04.wav', sr=44100)
y05, fs = librosa.load('./Passaro/P05.wav', sr=44100)
y06, fs = librosa.load('./Passaro/P06.wav', sr=44100)
y07, fs = librosa.load('./Passaro/P07.wav', sr=44100)
y08, fs = librosa.load('./Passaro/P08.wav', sr=44100)
y09, fs = librosa.load('./Passaro/P09.wav', sr=44100)
y10, fs = librosa.load('./Passaro/P10.wav', sr=44100)


d0 = librosa.stft(x01, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxM01, scxM01 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(x02, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxM02, scxM02 = np.mean(centX), np.std(centX)

d0 = librosa.stft(x03, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxM03, scxM03 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(x04, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxM04, scxM04 = np.mean(centX), np.std(centX)

d0 = librosa.stft(x05, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxM05, scxM05 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(x06, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxM06, scxM06 = np.mean(centX), np.std(centX)

d0 = librosa.stft(x07, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxM07, scxM07 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(x08, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxM08, scxM08 = np.mean(centX), np.std(centX)

d0 = librosa.stft(x09, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxM09, scxM09 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(x10, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxM10, scxM10 = np.mean(centX), np.std(centX)




d0 = librosa.stft(y01, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxP01, scxP01 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(y02, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxP02, scxP02 = np.mean(centX), np.std(centX)

d0 = librosa.stft(y03, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxP03, scxP03 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(y04, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxP04, scxP04 = np.mean(centX), np.std(centX)

d0 = librosa.stft(y05, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxP05, scxP05 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(y06, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxP06, scxP06 = np.mean(centX), np.std(centX)

d0 = librosa.stft(y07, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxP07, scxP07 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(y08, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxP08, scxP08 = np.mean(centX), np.std(centX)

d0 = librosa.stft(y09, n_fft=2048, hop_length=128, win_length=1024, window='hann') #dft
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512) # centroide espectral
mcxP09, scxP09 = np.mean(centX), np.std(centX) # media

d0 = librosa.stft(y10, n_fft=2048, hop_length=128, win_length=1024, window='hann')
centX = librosa.feature.spectral_flatness(S=np.abs(d0), n_fft=2048,\
                hop_length=512)
mcxP10, scxP10 = np.mean(centX), np.std(centX)


means_martelo = np.mean ( np.array( [mcxM01, mcxM02, mcxM03, mcxM04, mcxM05, mcxM06, \
									 mcxM07, mcxM08, mcxM09, mcxM10] ))
stds_martelo =  np.std ( np.array( [mcxM01, mcxM02, mcxM03, mcxM04, mcxM05, mcxM06, \
									 mcxM07, mcxM08, mcxM09, mcxM10] ))
means_passaro = np.mean ( np.array( [mcxP01, mcxP02, mcxP03, mcxP04, mcxP05, mcxP06, \
									 mcxP07, mcxP08, mcxP09, mcxP10] ))
stds_passaro =  np.std ( np.array( [mcxP01, mcxP02, mcxP03, mcxP04, mcxP05, mcxP06, \
									 mcxP07, mcxP08, mcxP09, mcxP10] ))

#means_martelo = np.mean ( np.array( [scxM01, scxM02, scxM03, scxM04, scxM05, scxM06, scxM07, scxM08, scxM09, scxM10] ))
#stds_martelo =  np.std ( np.array( [scxM01, scxM02, scxM03, scxM04, scxM05, scxM06, scxM07, scxM08, scxM09, scxM10] ))
#means_passaro = np.mean ( np.array( [scxP01, scxP02, scxP03, scxP04, scxP05, scxP06, scxP07, scxP08, scxP09, scxP10] ))
#stds_passaro =  np.std ( np.array( [scxP01, scxP02, scxP03, scxP04, scxP05, scxP06, scxP07, scxP08, scxP09, scxP10] ))


t, p = st.ttest_ind_from_stats(means_martelo, stds_martelo, 10,\
                               means_passaro, stds_passaro, 10)
print(means_martelo, stds_martelo)
print(means_passaro, stds_passaro)
print("P-value", p)

