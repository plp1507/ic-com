"""
Simulação de Montecarlo para sistemas BPSK OFDM através de canal PLC

Autor: Pedro Lucca
Data: 03/06/2024
Versão: 1
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from scipy.special import erfc


#%% Funções
def DMTmap(inpt):
    """
    Função para o mapeamento DMT

    Parâmetros
    ----------
    inpt: Mensagem a ser mapeada.
    
    Saída
    -----
    out: Mensagem após o mapeamento DMT.

    """
    dmt1 = np.vstack((inpt[-1], inpt[:N-1]))
    dmt2 = np.vstack((inpt[-1], np.flipud(inpt[:N-1])))
    dmtmap = np.vstack((dmt1, dmt2))
    return dmtmap

def DMTdemap(inpt):
    out = np.vstack((inpt[1:N], inpt[0]))
    return out


#%% Parâmetros do sistema
# Número de subportadoras
N = 10**3
# Número de blocos de símbolo
L = 64
# Comprimento do prefíxo cíclico
Ncp = 10
#Limites da SNR (dB)
snr_min = 0
snr_max = 9

# SNR (dB)
npt = 5
snr_dB = np.linspace(snr_min, snr_max, npt)
# SNR linear
snr = 10**(snr_dB/10)
# Potência do ruído
sigma_2 = 1/snr

# teorico
snr_dB_teorico = np.linspace(snr_min, snr_max, 1000)
ser_teorica = (1/2)*erfc(np.sqrt(10**(snr_dB_teorico/10)))

#%% Carregamento do arquivo referente ao canal PLC
canal = loadmat("../NB_0_500k.mat")
canal = canal['h'][0]
canal = np.concatenate((canal, np.zeros((2*N+Ncp)*L - len(canal))))

#%% Geração do sinal OFDM
#escolha aleatória de pontos da constelação, mundança serie/paralelo e mapeamento DMT
X = 2*np.round(np.random.rand(N*L)) - 1
Xl = np.reshape(X, [N, L], order = 'F')
Xmap = DMTmap(Xl)

#aplicação da IDFT, adição do prefx. cíclico e conversão paralelo/serie
x = np.fft.ifft(Xmap, axis = 0, norm = 'ortho')
xcp = np.vstack((x[2*N-Ncp:], x))
xn = np.reshape(xcp, -1, order = 'F')

#convolução com a resposta ao impulso do canal
y_tilde = np.convolve(xn.real,canal)[:(2*N + Ncp)*L]

#%% Simulação Montecarlo
ser = np.zeros(len(snr))

for k, Noise in enumerate(sigma_2):
    #geração do ruído a partir da SNR
    v = np.sqrt(Noise)*np.random.randn(len(y_tilde))

    #checagem da SNR obtida
    Pv = np.sum(np.abs(v)**2)/len(v)
    
    #adição de ruído
    y = y_tilde + v
    
    y = np.fft.ifft(np.fft.fft(y)/np.fft.fft(canal))
    
    #conversão serie/paralelo, remoção do prefx. cíclico, aplicação da DFT e desmapeamento
    xcpR = np.reshape(y, [2*N+Ncp, L], order = 'F')
    xR = np.delete(xcpR, range(Ncp), axis = 0)
    XlR = np.fft.fft(xR, axis = 0, norm = 'ortho').real
    Xdemap = DMTdemap(XlR)
    
    #conversão paralelo/série
    XR = np.reshape(Xdemap, -1, order = 'F')

    #contagem do erro e SER
    ser[k] = np.sum(np.sign(XR) != X)/(N*L)
   
    print('SNR = {} dB'.format(snr_dB[k]))
    print('SNR simulada = {} dB'.format(10*np.log10(1/Pv)))
    
    print('SER obtida = {}'.format(ser[k]))
    print("")


#%% Figuras
#SNRdB x SER teórica
plt.semilogy(snr_dB_teorico, ser_teorica, label ='curva teórica')
plt.semilogy(snr_dB, ser, 'o:', label ='pontos simulados')

plt.ylabel('SER')
plt.xlabel('SNR (dB)')

plt.legend()
plt.grid()
plt.show()

