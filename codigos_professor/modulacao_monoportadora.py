# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:22:22 2025

@author: acamp
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import upfirdn

#%% Parâmetros de simulação
K = 10 # Número de símbolos
N = 21 # Comprimento do pulso de transmissão

# Pulso de transmissão: 1(banda passante) / 0... (banda base)
banda = 1

#%% Geração do sinal a ser transmitido
# Símbolos BPSK
s = 2*np.round(np.random.rand(K)) - 1

# Pulso de transmissão NRZ
p = np.ones(N)

# Normalizando a energia do pulso
p = p/np.linalg.norm(p)

# Geração do sinal a ser transmitido
s_up = np.zeros(K*N)
s_up[::N] = s
x = np.convolve(s_up, p)[:K*N]

if banda == 1:
    m = 4 # Índice da frequência normalizada
    xb = x
    x = xb * np.exp(1j*2*np.pi * m * np.arange(K*N) / N)
    
    # Figura
    fig,ax = plt.subplots(3,1,figsize=(8,6))
    ax[0].stem(s_up)
    ax[0].set_ylabel(r"$s_k$")
    ax[0].set_xlabel(r"$k$ (símbolos)")
    ax[0].grid()
    ax[1].plot(xb,'-k')
    ax[1].set_ylabel(r"$x_b[n]$")
    ax[1].set_xlabel(r"$n$ (amostras)")
    ax[1].grid()
    ax[2].plot(x,'-k')
    ax[2].set_ylabel(r"$x[n]$ (parte real)")
    ax[2].set_xlabel(r"$n$ (amostras)")
    ax[2].grid()
    plt.tight_layout()
    plt.show()    
else:
    # Figura
    fig,ax = plt.subplots(2,1,figsize=(8,4))
    ax[0].stem(s_up)
    ax[0].set_ylabel(r"$s_k$")
    ax[0].set_xlabel(r"$k$ (símbolos)")
    ax[0].grid()
    ax[1].plot(x,'-k')
    ax[1].set_ylabel(r"$x[n]$")
    ax[1].set_xlabel(r"$n$ (amostras)")
    ax[1].grid()
    plt.tight_layout()
    plt.show()

#%% Transmissão e recepção do sinal
# --------------------------------------------------
# Transmissão do sinal
# --------------------------------------------------

#  .........

# -------------------------------------------------
# Recepção do sinal
# -------------------------------------------------
# filtro casado
q = p[::-1]
# Saída do filtro casado
if banda == 1: 
    xb = x * np.exp(-1j*2*np.pi * m * np.arange(K*N) / N)
    z = np.convolve(xb,q)[:len(xb)]
else: 
    z = np.convolve(x,q)[:len(x)]

# Descontar o atraso do pulso NRZ (transmissão e recepção)
# Note que se o pulso é um filtro FIR de ordem N-1
atraso = int((N-1)/2)
z = z[2*atraso:]

# Figura
plt.figure(figsize=(8,4))
plt.stem(s_up, label=r'$s_k$')
plt.plot(z,'-k', label=r'$z[n]$')
plt.ylabel("Amplitude")
plt.xlabel(r"$n$ (amostras)")
plt.grid()
plt.legend(loc ='upper center',bbox_to_anchor=(0.46, 1.2), ncol=2, borderaxespad=0, frameon=False)
plt.show()

# Amostragem com período em amostras N, que é o intervalo de símbolo que nesse
# exemplo coincide com a duração do pulso (note que nem sempre será o caso)
z_k = z[::N]

# Estimação dos símbolos transmitidos
# A forma genérica é a distância euclidiana. Mas como o exemplo é simples,
# vamos manter, por enquanto, a estimação de forma simples
x_hat = np.sign(z_k)

# Número de erros
erros = sum(x_hat != s)

# Taxa de erro
ber = erros/K



