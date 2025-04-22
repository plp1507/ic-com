# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 13:16:45 2025

@author: Ândrei Camponogara
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn 
from scipy.fft import fft, fftshift

#%% Funções
def srrc(alpha, Tb, L, span, plot = False):
    '''
    Gera o pulso raiz quadrada do cosseno levantado.

    Entradas
    ----------
    alpha : Fator de roll-off
    Tb : Período de símbolo
    L : Fator de sobreamostragem
    span : Número de símbolos que o pulso se espalha
    plot : Se igual a 'True', o pulso gerado é plotado

    Saídas
    -------
    (t, p) : Base temporal t e o sinal p(t) como tupla    
    
    '''
    
    Ts = Tb/L
    
    t = np.arange(-span*Tb/2, span*Tb/2 + Ts, Ts)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.divide(np.sin(np.pi*t*(1-alpha)/Tb) + 
                      (4*alpha*t/Tb) * np.cos(np.pi*t*(1+alpha)/Tb), 
                      (np.pi*t/Tb)*(1 - (4*alpha*t/Tb)**2))
    
    p = (1/Tb) *  A
    
  
    # -----------------------------------    
    # Lidando com as singularidades
    # -----------------------------------
    
    # Singularidade em p(t = 0)
    p[np.argwhere(np.isnan(p))] = (1/Tb) * (1 - alpha + 4*alpha/np.pi) 
    # Singularidade em t = +/- Tb/(4*alpha)
    p[np.argwhere(np.isinf(p))] = ((alpha/(np.sqrt(2)*Tb))*((1+(2/np.pi))*
                                    np.sin(np.pi/(4*alpha)) + 
                                   (1-(2/np.pi))*np.cos(np.pi/(4*alpha))))
    
    # -----------------------------------
    # Atraso do filtro
    atraso = int(span * L / 2)
    if plot == True:
        # FFT do pulso de transmissão
        Lfft = 2**np.ceil(np.log2(len(t)) + 1).astype('int')*4
        P_f = fftshift(fft(p, Lfft))
        f = np.arange(-Lfft/2, Lfft/2)/(Lfft*Ts)
        
    
        # Plote do pulso de transmissão p(t)
        fig, ax = plt.subplots(1,2, figsize = (14,4))
        ax[0].plot(t, p, '#5050b7f1', lw = 4)
        ax[0].set_xlabel(r'$t$ [s]', fontsize = 14)
        ax[0].set_ylabel(r'$p(t)$', fontsize = 14)
        ax[0].set_title('Filtro SRRC espalhado em {} símbolos'.format(span), fontsize = 14)
        ax[0].grid()
        
        ax[1].plot(f, abs(P_f), '#5050b7f1', lw = 4)
        ax[1].set_xlabel(r'$f$ [Hz]', fontsize = 14)
        ax[1].set_ylabel(r'$|P(f)|$', fontsize = 14)
        ax[1].set_title('Resposta em frequência do filtro SRRC', fontsize = 14)
        ax[1].set_xlim(-4/Tb, 4/Tb)
        ax[1].grid()
        plt.show()
    
    return t, p, atraso

#%% Parâmetros de simulação
filtro = 'srrc'              # filtro de transmissão: nrz ou srrc
# NRZ  (N=M)
# SRRC  (N>M)
N = 12                      # Comprimento do bloco de símbolo
M = 10                      # Número de subportadoras
I = 10                      # Número de blocos de símbolo 
Np = 20                     # Intervalos de símbolo que o pulso dura

T = 1                       # Período de amostragem
Ts = N*T                    # Período de bloco de símbolo 

fk = np.arange(M)/(M*T)     # Frequências

#%% Geração de símbolos 4-QAM
# Constelação
a = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])*np.sqrt(2)/2
# M fluxo de símbolos
np.random.seed(21)
X_m = np.random.choice(a,(M,N*I))

#%% Pulso de transmissão com energia unitária



if filtro == 'nrz':
    # ------------------------------------
    # Retangular    
    t = np.arange(-Ts/2,Ts/2,T)
    pt = np.zeros((M,len(t)), dtype='complex')
    for k in range(len(fk)):
        pt[k] = np.ones(len(t))/np.sqrt(len(t)) * np.exp(1j*2*np.pi*fk[k]*t)
   
elif filtro == 'srrc':     
    # ------------------------------------
    # SRRC   
     
    # Fator de roll-off
    alpha = round(N/M-1,2)
    # Pulso raiz do cosseno levantado
    t, qt, atraso = srrc(alpha, Ts, N, Np, plot = True)
    
    pt = np.zeros((M,len(t)), dtype='complex')
    for k in range(len(fk)):
        pt[k] = qt * np.exp(1j*2*np.pi*fk[k]*t)
        # Normalização da energia
        pt[k] = pt[k]/np.sqrt(np.sum(np.abs(pt[k])**2))


#%% Modulação FMT
aux = upfirdn(pt[0], X_m[0], N)
x = np.zeros((M,len(aux)),dtype='complex')

for k in range(M):
    x[k] = upfirdn(pt[k], X_m[k], N)
    
xt = np.sum(x, axis=0)
    
#%% Demodulação FMT
# Filtro casado
z = np.zeros((M, len(xt) + len(pt[0])-1),dtype='complex')
for k in range(M):
    z[k] = np.convolve(np.conj(pt[k,::-1]), xt)
    plt.plot(z[k][2*atraso:-2*atraso])
    plt.plot(x[k][atraso:-atraso])
    plt.grid()
    plt.show()

zk = np.zeros((M, N*I), dtype='complex')
for k in range(M):
    if filtro == 'nrz':
        zk[k] = z[k,N-1::N]
    elif filtro == 'srrc':
        zk[k] = z[k,2*atraso:-2*atraso:N]
    
#%%
# Verificação

plt.figure()
plt.plot(X_m[3,:].real, 'o-', label='Tx real')
plt.plot(zk[3,:].real, 'x--', label='Rx real')
plt.title(f'Subportadora {3}')
plt.legend()
plt.grid(True)
    
plt.show()

 
   
