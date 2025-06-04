"""
Simulação de Montecarlo para sistemas BPSK OFDM

Autor: Pedro Lucca
Data: 03/06/2025
Versão: 1
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import erfc

#%% Características da SNR
npt = 10

SNRt = np.linspace(-2, 10, 1000)
SNRs = SNRt[::1000//npt - 1]

SERt = (1/2)*erfc(np.sqrt(10**(SNRt/10)))
SERs = np.zeros(npt)

sigma2 = 10**(-SNRs/10)

#%% Parâmetros do sistema
# Número de subportadoras
N = 64
# Número de blocos de símbolo
L = 10**4
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

print(np.mean(xn**2))

#%% Simulação Montecarlo
SERs = np.zeros(len(SNRs))

for k, Noise in enumerate(sigma2):
    #passagem pelo canal AWGN
    v = np.sqrt(Noise/2)*(np.random.randn((N+Ncp)*L) + 1j*np.random.randn((N+Ncp)*L))
    
    Pv = np.mean(np.abs(v)**2)
    
    xcpR = xn + v

    #conversão serie/paralelo, remoção do prefx. cíclico e aplicação da DFT
    xcpR = np.reshape(xcpR, [N+Ncp, L], order = 'F')
    xR = np.delete(xcpR, range(Ncp), axis=0)
    XlR = np.fft.fft(xR, axis = 0, norm = 'ortho').real

    #conversão paralelo/série
    XR = np.reshape(XlR, -1, order = 'F')

    #contagem do erro e BER
    SERs[k] = np.sum(np.sign(XR) != X)/(N*L)

    print(f'SNR teórica  = {round(SNRs[k], 3)} dB')
    print(f'SNR simulada = {round(10*np.log10(np.mean(XR**2)/Pv), 3)} dB\n')
    print(f'SER teórica  = {SERt[(1000//npt - 1)*k]}')
    print(f'SER simulada = {SERs[k]}\n\n')

#%% Figuras
plt.semilogy(SNRt, SERt, label ='curva teórica')
plt.semilogy(SNRs, SERs, 'o:', label ='pontos simulados')

plt.ylabel('SER')
plt.xlabel('SNR dB')

plt.legend()
plt.grid()
plt.show()
