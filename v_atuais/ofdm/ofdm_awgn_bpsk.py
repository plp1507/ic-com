"""
Simulação de Montecarlo para sistemas BPSK OFDM

Autor: Pedro Lucca
Data: 24/05/2024
Versão: 1.0
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from math import erfc

#%% Características da SNR
npt = 10

SNRt = np.linspace(-8, 8, 1000)
SNRs = SNRt[::1000//npt - 1]

SERt = (1/2)*erfc(np.sqrt(10**(SNRt/10)))
SERs = np.zeros(npt)

sigma2 = 10**(-SNRs/10)

#%% Parâmetros do sistema
# Número de subportadoras
N = 128
# Número de blocos de símbolo
L = 1000
# Comprimento do prefíxo cíclico
Ncp = 0

#%% Geração do sinal OFDM
#escolha aleatória de pontos da constelação, mundança serie/paralelo
X = 2*np.round(np.random.rand(N*L)) - 1
Xl = np.reshape(X, [N, L], order = 'F')

#aplicação da IDFT, adição do prefx. cíclico e conversão paralelo/serie
x = np.fft.ifft(Xl, axis = 0, norm = 'ortho')
xcp = np.vstack((x[N-Ncp:], x))
xn = np.reshape(xcp,((N+Ncp)*L), order = 'F')

#%% Simulação Montecarlo
ser = np.zeros(len(snr))

for k,Noise in enumerate(sigma_2):
    #passagem pelo canal AWGN
    v = np.sqrt(Noise)*(np.random.randn(len(xn)))
    
    Pv = np.sum(np.abs(v)**2)/len(v)
    
    xcpR = xn + v

    #conversão serie/paralelo, remoção do prefx. cíclico e aplicação da DFT
    xcpR = np.reshape(xcpR, [N+Ncp, L], order = 'F')
    xR = np.delete(xcpR, range(Ncp), axis=0)
    XlR = np.fft.fft(xR, axis = 0, norm = 'ortho')

    #conversão paralelo/série
    XR = np.reshape(XlR, -1, order = 'F')

    #detecção dos pontos após o ruído
    XRd = np.zeros(len(XR), dtype = "complex_")
    for i in range(N*L):
        XRd[i] = np.sign(XR[i])
    
    #contagem do erro e BER
    erro = np.sum(XRd != X)
    ser[k]  = erro/(N*L)
    
    print('SNR = {} dB'.format(snr_dB[k]))
    print('SNR simulada = {} dB'.format(10*np.log10(1/Pv)))
    
    plt.figure(figsize=(4,4))
    plt.scatter(XR.real, XR.imag)
    plt.scatter(constel.real, constel.imag)   
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.show()

#%% Figuras
#SNR x SER teórica
plt.figure(figsize=(6,3))
plt.plot(, ser_teorica, label ='curva teórica')

# SNR simulada x SER
plt.plot(snr_dB, ser, 'o', label ='pontos simulados')

plt.ylabel('SER')
plt.xlabel('SNR dB')

plt.yscale('log')
plt.ylim(10**-6,1)

plt.grid()
plt.show()
