# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:22:22 2025

@author: acamp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
#from scipy.signal import upfirdn

#%% Funções
def Q(x):
    """
    Função Q.

    """
    return 0.5 - 0.5*erf(x/np.sqrt(2))

#%% Parâmetros de simulação
K = 10**5 # Número de símbolos de constelação digital
N = 21 # Intervalo de símbolo (monoportadora) ou bloco de símbolo (multiportadora)

snr = np.arange(0,25,5) # SNR em dB 
sigma2 = 1/10**(snr/10) # Potência do ruído (constelação precisa ter potência 1)

# Pulso de transmissão: 1(banda passante) / 0... (banda base)
banda = 1

#%% Geração do sinal a ser transmitido
# Símbolos QPSK
si = 2*np.round(np.random.rand(K)) - 1
sq = 2*np.round(np.random.rand(K)) - 1
s = si + 1j*sq

# Normalização da potência dos simbolos gerados
s_norm = s/np.sqrt(np.mean(abs(s)**2))

# Pulso de transmissão NRZ
p = np.ones(N)

# Normalizando a energia do pulso
p /= np.linalg.norm(p)

# filtro casado
q = p[::-1]

# Geração do sinal a ser transmitido
s_up = np.zeros(K*N, dtype='complex')
s_up[::N] = s_norm
x = np.convolve(s_up, p)[:K*N]

if banda == 1:
    m = 4 # Índice da frequência normalizada
    M = N 
    fm = m/M # Frequência normalizada
    x *= np.exp(1j*2*np.pi * fm * np.arange(len(x)))

#%% Transmissão e recepção do sinal
ber = np.zeros(len(sigma2))

for ind,val in enumerate(sigma2):
    # --------------------------------------------------
    # Transmissão do sinal
    # --------------------------------------------------
    # Geração do ruído
    v = np.sqrt(val/2)*(np.random.randn(len(x)) + 1j*np.random.randn(len(x)))
    # Sinal na entrada do receptor 
    y = x + v
    
    # -------------------------------------------------
    # Recepção do sinal
    # -------------------------------------------------
    # Saída do filtro casado
    if banda == 1: 
        yb = y * np.exp(-1j*2*np.pi * fm * np.arange(len(y)))
        z = np.convolve(yb,q)[:len(yb)]
    else: 
        z = np.convolve(y,q)[:len(y)]
    
    # Descontar o atraso do pulso NRZ (transmissão e recepção)
    # Note que se o pulso é um filtro FIR de ordem N-1
    atraso = int((N-1)/2)
    z = z[2*atraso:]
    
    # Amostragem com período em amostras N, que é o intervalo de símbolo que nesse
    # exemplo coincide com a duração do pulso (note que nem sempre será o caso)
    z_k = z[::N]
    
    print('SNR real = {} dB'.format(10*np.log10(1/val)))
    print('SNR simulada = {} dB'.format(10*np.log10(np.mean(abs(z_k)**2)/np.mean(abs(v)**2))))    
    
    # Estimação dos símbolos transmitidos
    # A forma genérica é a distância euclidiana. Mas como o exemplo é simples,
    # vamos manter, por enquanto, a estimação de forma simples
    x_hat = np.sign(np.real(z_k)) + 1j*np.sign(np.imag(z_k))
    
    # Número de erros
    erros = sum(x_hat != s)
    print("Número de erros = {}".format(erros))
    
    # Taxa de erro
    ber[ind] = erros/(2*K)
            
# Probabilidade de erro (BER Teórica)
snr_t = np.linspace(0,20,500)
Pe = Q(np.sqrt(10**(snr_t/10)))
    
# Figura
plt.figure(figsize=(6,4))
plt.semilogy(snr,ber,'ok', label='Monte Carlo')
plt.semilogy(snr_t,Pe,'-k', label='Teórica')
plt.ylabel("BER")
plt.xlabel("SNR (dB)")
plt.grid()
plt.ylim(10**-6,1)
plt.legend(loc ='upper center',bbox_to_anchor=(0.46, 1.2), ncol=2, borderaxespad=0, frameon=False)
plt.show()
