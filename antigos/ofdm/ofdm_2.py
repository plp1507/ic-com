"""
Simulação de Montecarlo para sistemas QAM OFDM

Autor: Pedro Lucca
Data: 24/09/2024
Versão: 2
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

#%% Funções
def constellation(m):
    '''
    Função para gerar a constelação como vetor de tamanho mPAM


    Parâmetros
    ----------
    m : Ordem da constelação.

    Saídas
    -------
    out : símbolos de uma constelação m-PAM.
    '''
    range_ = int(np.sqrt(m))
    out = np.zeros([range_,range_], dtype = "complex_")
    for i in range(range_):
        for j in range(range_):
            out[i][j] = 1*i + 1j*j
    
    out = np.reshape(out, m)

    #p/ centrar a const. em 0
    out -= (np.sqrt(m)-1)*(0.5 +0.5j)

    #normalização pela energia média da constelação
    energTotal = np.sum(np.abs(out)**2)/m
    out /= np.sqrt(energTotal)
    
    return out

def detec(msgR):
    """
    Função de detecção de símbolos

    Parameters
    ----------
    msgR : Símbolo a ser detectado

    Returns
    -------
    Ponto da constelação mais próximo

    """
    for i in range(M):
        XA = np.column_stack((msgR.real, msgR.imag))
        XB = np.column_stack((constel.real, constel.imag))
        distcs = cdist(XA, XB, metric = 'euclidean')
    return constel[np.argmin(distcs)]

#%% Parâmetros do sistema
# Número de subportadoras
N = 128
# Número de blocos de símbolo
L = 1000
# Comprimento do prefíxo cíclico
Ncp = 0
# Ordem da constelação
M = 16

# SNR (dB)
snr_dB = np.arange(0,35,5)
# SNR linear
snr = 10**(snr_dB/10)
# Potência do ruído
sigma_2 = 1/snr

#%% Geração do sinal OFDM
# Geração da consteção M-PAM
constel = constellation(M)

#escolha aleatória de pontos da constelação, mundança serie/paralelo
X = np.random.choice(constel, N*L)
Xl = np.reshape(X, [N, L], order = 'F')

#aplicação da IDFT, adição do prefx. cíclico e conversão paralelo/serie
x = np.fft.ifft(Xl, axis = 0, norm = 'ortho')
xcp = np.vstack((x[N-Ncp:], x))
xn = np.reshape(xcp,((N+Ncp)*L), order = 'F')

#%% Simulação Montecarlo
ber = np.zeros(len(snr))

for k,Noise in enumerate(sigma_2):
    print('SNR = {} dB'.format(snr_dB[k]))
    #passagem pelo canal AWGN
    v = np.sqrt(Noise/2)*(np.random.randn(len(xn)) + 1j*np.random.randn(len(xn)))
    
    Pv = np.sum(np.abs(v)**2)/len(v)
    
    print('SNR simulada = {} dB'.format(10*np.log10(1/Pv)))
        
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
        XRd[i] = detec(XR[i])
    
    #contagem do erro e BER
    erro = np.sum(XRd != X)
    ber[k]  = erro/(N*L)
    
    plt.figure(figsize=(4,4))
    plt.scatter(XR.real, XR.imag)
    plt.scatter(constel.real, constel.imag)   
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.show()


#%% Figuras
#SNR x SER teórica
plt.figure(figsize=(6,3))
plt.plot(snr_dB_teorico, ser_teorica))

# SNR simulada x SER
plt.plot(snr_dB, ser, 'o')
plt.yscale('log')
plt.ylim(10**-6,1)

plt.grid()
plt.show()
