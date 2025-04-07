"""
Simulação de Monte Carlo de sistema FMT - QAM

Autor: Pedro Lucca Pereira
Data:  28/10/2025
"""

import numpy as np
from matplotlib import pyplot as plt
from math import erfc
from scipy.spatial.distance import cdist

#%% Funções
def decide(msg, constell):
    '''
    Função de decisão (hard decide) de símbolos
    Parâmetros de entrada:
        msg: símbolos recebidos
        constell: constelação a ser utilizada

    Saída:
        entrada 'msg' corrigida para os pontos da constelação mais próximos

    '''

    XA = np.column_stack((msg.real, msg.imag))
    XB = np.column_stack((constell.real, constell.imag))
    distcs = cdist(XA, XB, metric = 'euclidean')

    return constell[np.argmin(distcs, axis = 1)]

def qam_constel(ordem):
    '''
    Função de geração da constelação ordem-QAM
    Parâmetros de entrada:
        ordem: Ordem da constelação

    Saída:
        vetor contendo os símbolos normalizados da constelação QAM

    '''

    range_ = int(np.sqrt(ordem))
    out = np.zeros([range_, range_], dtype = 'complex')
    for i in range(range_):
        for j in range(range_):
            out[i][j] = i + 1j*j

    out = np.reshape(out, ordem)
    out -= (np.sqrt(ordem)-1)*(.5 + .5j)
    e_total = np.sum(np.abs(out)**2)/ordem
    out /= np.sqrt(e_total)
    return out

#%% Configurações da SNR
SNRteorico = np.linspace(0, 20, 1000)
SNRsimulado = SNRteorico[::len(SNRteorico)//7]
sigma2 = 10**(-SNRsimulado/10)

#%% Simulação
#Número de símbolos, comprimento dos blocos,
#número de portadoras e ordem da constelação
n = 10**6
l = 11
n_portadoras = 5
I = 16

#Formato dos pulsos
p = np.ones(l); p /= np.linalg.norm(p)

#Geração dos símbolos aleatórios I-QAM com n_port portadoras
constelacao = qam_constel(I)
mensagem = np.random.choice(constelacao, n*n_portadoras)
seq = np.reshape(mensagem, [n_portadoras, n])

msg_ups = np.zeros([n_portadoras, l*n], dtype = 'complex')
x = np.zeros([n_portadoras, l*n], dtype = 'complex')

for i in range(n_portadoras):
    #upsampling
    msg_ups[i][::l] = seq[i]
    #passagem pelo filtro
    x[i] = np.convolve(msg_ups[i], p)[:l*n]

#modulação do sinal
xb = x
for i in range(n_portadoras):
    x[i] = xb[i] * np.exp(2j*np.pi*(4+i)*np.arange(l*n)/l)

x = np.sum(x, axis=0)

#recepção do sinal
#filtro casado
q = p[::-1]

z = np.zeros([n_portadoras, l*n], dtype = 'complex')
z_d = np.zeros([n_portadoras, n], dtype = 'complex')
fim = np.zeros(n*n_portadoras, dtype = 'complex')

SERsimulado = np.zeros(len(SNRsimulado))

for k, noise in enumerate(sigma2):

    w = np.sqrt(noise/2)*(np.random.randn(n*l) + 1j*np.random.randn(n*l))

    y = x + w

    for i in range(n_portadoras):
        xb[i] = y*np.exp(-2j*np.pi*(4+i)*np.arange(n*l)/l)
        z[i] = np.convolve(xb[i], q)[:l*n]
        #remoção do atraso de grupo e downsampling
        z_d[i] = z[i][2*int((l-1)/2)::l]

    fin = decide(np.reshape(z_d, -1), constelacao)
    SERsimulado[k] = np.sum(fin != mensagem)/(n*n_portadoras)
    
    #Checagem
    print(f'SNR: {SNRsimulado[k]}dB')
    print(f'SNR calculada: {10*np.log10(np.mean(abs(fin)**2)/np.mean(abs(w)**2))}dB')
    print(f'SER obtida: {SERsimulado[k]}\n')

#%% Geração da curva teórica
SERteorico = np.zeros(len(SNRteorico))

for i in range(len(SNRteorico)):
    argerfc = (3/(2*(I-1)))*(10**(SNRteorico[i]/10))
    p = (1-np.sqrt(1/I))*erfc(np.sqrt(argerfc))
    SERteorico[i] = 1 - ((1-p)**2)

#%% Plots dos pontos
plt.semilogy(SNRteorico, SERteorico, label = 'curva teórica')
plt.semilogy(SNRsimulado, SERsimulado, 'o:', label = 'pontos simulados')

plt.xlabel('SNR (dB)')
plt.ylabel('SER')

plt.title(f'Taxa de erro de símbolo de sistema {I}QAM com {n_portadoras} portadoras')

plt.grid()
plt.legend()
plt.show()
