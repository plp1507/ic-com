'''
Funcionamento de sistema FMT
Autor: Pedro Lucca Pereira
Data: 29/12/2024
Versão: 1
'''

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from math import pi

def dec(msg, constel):
    res = np.zeros(np.shape(msg), dtype = 'complex')
    XA = np.column_stack((constel.real, constel.imag))
    for i in range(len(msg)):
        XB = np.column_stack((msg[i].real, msg[i].imag))
        distcs = cdist(XA, XB, 'euclidean')
        res[i] = constel[np.argmin(distcs)]

    return res

#comprimento do bloco multiportadora
#número de subportadoras
n = 10
m = 10

#constelação BPSK
constelacao = np.array([-1., -0.5, 0.5, 1.])


#escolha aleatória de símbolos
mensagem = np.random.choice(constelacao, m*n)
mensagem = np.reshape(mensagem, [m, n])


#upsampling
k = 20

mens_ups = np.zeros([m, k*n])
for i in range(m):
    mens_ups[i][::k] = mensagem[i]


#modulação da janela
janela = signal.get_window('boxcar', k)

janela_mod = np.zeros([m, k], dtype = 'complex')
for i in range(m):
    janela_mod[i] = janela.astype('complex')*np.exp(2j*pi*np.arange(k)*(i+1)/m)


#passagem pelo filtro
bloco_janela = np.zeros([m, n*k + k - 1], dtype = 'complex')
for i in range(m):
    bloco_janela[i] = np.convolve(mens_ups[i], janela_mod[i])

x = np.sum(bloco_janela, axis = 0)

#adição de ruído
sigma_2 = 0.0001
w = np.sqrt(sigma_2/2)*np.random.randn(k*n + k -1)
x += w

#passagem pelo filtro casado
bloco_casado = np.zeros([m, k*n + 2*k - 2], dtype = 'complex')
for i in range(m):
    bloco_casado[i] = np.convolve(x, np.conjugate(np.flip(janela_mod[i])))
    bloco_casado[i] /= np.max(bloco_casado[i])


#downsampling
mens_dws = np.zeros([m, n], dtype = 'complex')
for i in range(m):
    mens_dws[i] = np.delete(bloco_casado[i][k-1::k], -1)

mens_fin = np.reshape(mens_dws, -1)
mens_cor = dec(mens_fin, constelacao)


#configurações do plot
fig, axs = plt.subplots(2)

axs[0].stem(mensagem[0], label = 'mensagem')
axs[0].plot(np.linspace(0, n, n*k, endpoint = False), mens_ups[0], '.:', label = 'm. upsampling')
axs[0].plot(np.linspace(0, (n*k + k - 1)/k, n*k + k - 1, endpoint = False), np.abs(bloco_janela[0]), '.:', label = 'bloco apos janelamento')
axs[0].plot(np.linspace(0, n + 2 - 2/k, n*k + 2*k - 2, endpoint = False), np.abs(bloco_casado[0]),'o:', label = 'bloco apos filtro c.')
axs[0].plot(mens_dws[0].real, 'o:', label = 'm. downsampling')
axs[0].plot(mens_cor[:n].real, 'o:', label = 'm. corrigida')

axs[0].grid()
axs[0].legend()

axs[1].stem(mensagem[1], label = 'mensagem')
axs[1].plot(np.linspace(0, n, n*k, endpoint = False), mens_ups[1], '.:', label = 'm. upsampling')
axs[1].plot(np.linspace(0, (n*k + k - 1)/k, n*k + k - 1, endpoint = False), np.abs(bloco_janela[1]), '.:', label = 'bloco apos janelamento')
axs[1].plot(np.linspace(0, n + 2 - 2/k, n*k + 2*k - 2, endpoint = False), np.abs(bloco_casado[1]),'o:', label = 'bloco apos filtro c.')
axs[1].plot(mens_dws[1].real, 'o:', label = 'm. downsampling')
axs[1].plot(mens_cor[n:2*n].real, 'o:', label = 'm. corrigida')

axs[1].grid()
axs[1].legend()

plt.show()

