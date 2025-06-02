'''
Simulação de Montecarlo de sistema FMT
Autor: Pedro Lucca Pereira
Data: 31/12/2024
Versão: 1.1
'''
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from math import pi, erfc, log10

def qam(I):
    '''
    Função para gerar uma constelação QAM de ordem I
    
    Inputs:
        I - ordem da constelação QAM

    Outputs:
    Array contendo os símbolos da constelação
    '''
    out = np.zeros([int(np.sqrt(I)), int(np.sqrt(I))], dtype = 'complex')
    for i in range(int(np.sqrt(I))):
        for j in range(int(np.sqrt(I))):
            out[i][j] = i + j*1j
    
    out = np.reshape(out, -1)
    out -= (np.sqrt(I)-1)*(0.5+0.5j)
    et = np.sum(np.abs(out)**2)/I
    out /= np.sqrt(et)
    return out

def dec(msg, constel):
    '''
    Função de decisão e correção de símbolos
    
    Inputs:
        msg     - Símbolo de entrada a ser corrigido para a const. de referência
        constel - constelação de referência
    
    Outputs:
        símbolo corrigido
    '''
    res = np.zeros(np.shape(msg), dtype = 'complex')
    XA = np.column_stack((constel.real, constel.imag))
    for i in range(len(msg)):
        XB = np.column_stack((msg[i].real, msg[i].imag))
        distcs = cdist(XA, XB, 'euclidean')
        res[i] = constel[np.argmin(distcs)]

    return res


#Parâmetros da SNR
SNRmn = -20
SNRmx = 15
npt = 15

SNRdB = np.linspace(SNRmn, SNRmx, npt)
sigma_2 = 10**(-SNRdB/10)

#Comprimento do bloco multiportadora;
#Número de portadoras
#Ordem da constelação
n = 10000
m = 7
I = 4

#Geração da constelação QAM
constelacao = qam(I)


#escolha aleatória de símbolos
mensagem = np.random.choice(constelacao, m*n)
mens_par = np.reshape(mensagem, [m, n])


#Upsampling
#Fator de upsampling
k = 20

mens_ups = np.zeros([m, k*n], dtype = 'complex')
for i in range(m):
    mens_ups[i][::k] = mens_par[i]


#Modulação da janela
janela = signal.get_window('boxcar', k)

janela_mod = np.zeros([m, k], dtype = 'complex')
for i in range(m):
    janela_mod[i] = janela.astype('complex')*np.exp(2j*pi*np.arange(k)*(i+1)/m)


#Passagem pelo filtro
bloco_janela = np.zeros([m, n*k + k - 1], dtype = 'complex')
for i in range(m):
    bloco_janela[i] = np.convolve(mens_ups[i], janela_mod[i])


#Soma das portadoras
x = np.sum(bloco_janela, axis = 0)


#Simulação de Montecarlo
SER = np.zeros(len(SNRdB))

for l, noise in enumerate(sigma_2):
    #Geração do ruído
    w = np.sqrt(noise/2)*(np.random.randn(k*n + k -1) + np.random.randn(k*n + k - 1)*1j)
    print(f"SNR (dB): {SNRdB[l]}")
    
    #Checagem da SNR obtida
    Pv = np.sum(np.abs(w)**2)/len(w)
    print(f"SNR simulada (dB): {10*log10(1/Pv)}")


    #Adição do ruído
    y = x + w


    #Passagem pelo filtro casado
    bloco_casado = np.zeros([m, k*n + 2*k - 2], dtype = 'complex')
    for i in range(m):
        bloco_casado[i] = np.convolve(y, np.conjugate(np.flip(janela_mod[i])))


    #Downsampling
    #k-1 -> atraso de grupo dos dois filtros aplicados
    mens_dws = np.zeros([m, n], dtype = 'complex')
    for i in range(m):
        mens_dws[i] = np.delete(bloco_casado[i][k-1::k], -1)

    #Correção dos símbolos recebidos
    mens_dec = dec(np.reshape(mens_dws, -1), constelacao)
    
    #Contabilização dos erros e cálculo da SER
    SER[l] = np.sum(mens_dec!=mensagem)/(m*n)
    print(f"SER obtida: {SER[l]}\n")


#Obtenção da curva teórica
SERt = np.zeros(1000)
SNRdBt = np.linspace(SNRmn, SNRmx, 1000)

SNRl = 10**(SNRdBt/10)

for i in range(1000):
    argerfc = (3/(2*(I-1)))*SNRl[i]
    p = (1-np.sqrt(1/I))*erfc(np.sqrt(argerfc))
    SERt[i] = 1 - ((1 - p)**2)


#Configurações do plot
plt.plot(SNRdBt, SERt, label = 'curva teórica')
plt.plot(SNRdB, SER, '.:', label = 'pontos simulados')

plt.grid()
plt.semilogy()
plt.legend()
plt.show()

