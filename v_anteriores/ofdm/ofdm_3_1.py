"""
Simulação de Montecarlo para sistemas QAM OFDM através de canal PLC

Autor: Pedro Lucca
Data: 01/10/2024
Versão: 3
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from math import erfc

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
    out = np.zeros([range_,range_], dtype = "complex")
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
N = 12
# Número de blocos de símbolo
L = 100
# Comprimento do prefíxo cíclico
Ncp = 12
# Ordem da constelação
M = 16
#SNR máxima (dB)
snr_max = 90

# SNR (dB)
snr_dB = np.arange(50, snr_max, 3)
# SNR linear
snr = 10**(snr_dB/10)
# Potência do ruído
sigma_2 = 1/snr


#%%Curva teórica da SER
'''
snr_dB_teorico = np.linspace(0, snr_max, 1000)
snrT = 10**(snr_dB_teorico/10)
ser_teorica = np.zeros(len(snr_dB_teorico))
for i in range(len(snr_dB_teorico)):
    argerfc = (3/(2*(M-1)))*(snrT[i])
    p = (1-np.sqrt(1/M))*erfc(np.sqrt(argerfc))
    ser_teorica[i] = 1-((1-p)**2)
'''

#%% Carregamento do arquivo referente ao canal PLC
canal = loadmat('NB_0_500k.mat')
canal = canal['h'][0]
#canal = np.concatenate((canal, np.zeros((N+Ncp)*L - len(canal))))
canal = canal.astype("complex")

Dh = np.diag(canal)
ch_inv = np.diag(np.linalg.inv(Dh))

#%% Geração do sinal OFDM
# Geração da consteção M-PAM
constel = constellation(M)

#escolha aleatória de pontos da constelação, mundança serie/paralelo
X = np.random.choice(constel, N*L)
Xl = np.reshape(X, [N, L], order = 'F')

#aplicação da IDFT, adição do prefx. cíclico e conversão paralelo/serie
x = np.fft.ifft(Xl, axis = 0, norm = 'ortho')
xcp = np.vstack((x[N-Ncp:], x[:]))

xn = np.reshape(xcp,((N+Ncp)*L), order = 'F')


#%% Simulação Montecarlo
ser = np.zeros(len(snr))

for k, Noise in enumerate(sigma_2):
    #geração do ruído a partir da SNR
    v = np.sqrt(Noise/2)*(np.random.randn(len(xn)) + 1j*np.random.randn(len(xn)))
    v_ch = np.convolve(ch_inv, v, mode='same')

    #checagem da SNR obtida
    Pv = np.sum(np.abs(v)**2)/len(v)
    
    #adição de ruído
    xcpR = xn + v_ch
    
    #conversão serie/paralelo, remoção do prefx. cíclico e aplicação da DFT
    xcpR = np.reshape(xcpR, [N+Ncp, L], order = 'F')
    xR = np.delete(xcpR, range(Ncp), axis = 0)
    XlR = np.fft.fft(xR, axis = 0, norm = 'ortho')

    #conversão paralelo/série
    XR = np.reshape(XlR, -1, order = 'F')

    #detecção dos pontos após a passagem pelo canal
    XRd = np.zeros(len(XR), dtype = "complex")
    for i in range(N*L):
        XRd[i] = detec(XR[i])
    
    #contagem do erro e SER
    erro = np.sum(XRd != X)
    ser[k]  = erro/(N*L)
   
    print('SNR = {} dB'.format(snr_dB[k]))
    print('SNR simulada = {} dB'.format(10*np.log10(1/Pv)))
    
    print('SER obtida = {}dB'.format(ser[k]))
    print("")


#%% Figuras
#SNRdB x SER teórica
'''
plt.figure(figsize=(6,3))
plt.plot(snr_dB_teorico, ser_teorica, label ='curva teórica')
'''
# SNRdB x SER simulada
plt.plot(snr_dB, ser, 'o', label ='pontos simulados')

plt.ylabel('SER')
plt.xlabel('SNR (dB)')

plt.yscale('log')
plt.ylim(10**-6,1)

plt.grid()
plt.show()
