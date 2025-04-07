'''
Geração de símbolos BPSK em sistema FMT utilizando filtro RRC
Autor: Pedro L.
Data: 27/03/2025
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc

### configuração do filtro RRC

alpha = 0.5    ####deve ser maior que 0.2
k = 10        #fator de oversampling
Fs = 100     #frequência de amostragem (amostras/s)
Ts = k/Fs      #período de símbolo (s)
l = 10          #comprimento do filtro (segundos)

x = np.linspace(-l/2, l/2, l*Fs+1)/Ts
filtro_rrc = (1/Ts)*(np.sin(np.pi*x*(1-alpha)) + (4*alpha*x)*np.cos(np.pi*x*(1+alpha)))

filtro_rrc[x==0] = (1/Ts)*(1+alpha*(4/np.pi - 1))
filtro_rrc[abs(x) == 0.25*Ts/alpha] = (alpha/(Ts*np.sqrt(2)))*((1+2/np.pi)*np.sin(0.25*np.pi/alpha)+ (1-2/np.pi)*np.cos(0.25*np.pi/alpha))
filtro_rrc[x!=0] /= np.pi*x[x!=0]*(1-(4*alpha*x[x!=0])**2)

filtro_rrc /= np.sqrt(np.sum(filtro_rrc**2))

### configuração da mensagem
n = 10**4    #número de símbolos
m = 3        #número de portadoras

#m máximo: 5

#geração de símbolos bpsk
mensagem = 2*np.round(np.random.rand(n*m)) - 1
m_paralelo = np.reshape(mensagem, [m, n])

m_upsp = np.zeros([m, n*k])
m_conv = np.zeros([m, n*k + l*Fs])
m_modl = np.zeros([m, n*k + l*Fs], dtype = 'complex')

#####processamento da transmissão
### upsampling, convolução e modulação
for i in range(m):
    m_upsp[i][::k] = m_paralelo[i]
    m_conv[i] = np.convolve(m_upsp[i], filtro_rrc)
    m_modl[i] = m_conv[i]*np.exp(2j*np.pi*(i + m)*np.arange(n*k + Fs*l)/m)

sinal_tx = np.sum(m_modl, axis=0)

#####recepção
###demodulação, passagem por filtro casado e downsampling
m_demod = np.zeros([m, n*k + Fs*l], dtype= 'complex')
m_casad = np.zeros([m, n*k + 2*Fs*l], dtype = 'complex')
m_dwnsp = np.zeros([m, n], dtype = 'complex')

#####Configurações da SNR
npt_sim = 8

SNRt = np.linspace(0, 10, 1000)
SNRs = SNRt[::1000//(npt_sim) - 1]
sigma2 = 10**(-SNRs/10)
SERt = 0.5*erfc(np.sqrt(10**(SNRt/10)))
SERs = np.zeros(len(SNRs))

for j, noise in enumerate(sigma2):
    w = np.sqrt(noise/2)*np.random.randn(n*k + Fs*l)
    
    sinal_rx = sinal_tx + w

    for i in range(m):
        m_demod[i] = sinal_rx*np.exp(-2j*np.pi*(i + m)*np.arange(n*k + Fs*l)/m)
        m_casad[i] = np.convolve(m_demod[i], filtro_rrc)
        m_dwnsp[i] = m_casad[i][l*Fs:l*Fs + n*k:k]
 
    plt.stem(m_dwnsp[0][:10])
    plt.stem(mensagem[:10])
    plt.grid()
    plt.show()

    m_reconstruida = np.reshape(np.sign(m_dwnsp.real), -1)
    SERs[j] = np.sum(m_reconstruida != mensagem)/(m*n)

    print(f'SNR simulada: {10*np.log10(.5*np.mean(abs(m_reconstruida)**2)/np.mean(abs(w)**2))}dB')
    print(f'SNR teórica : {SNRs[j]}dB\n')
    print(f'SER teórica : {SERt[::1000//(npt_sim) - 1][j]}')
    print(f'SER obtida  : {SERs[j]}\n\n')



plt.title(f'Sistema FMT-BPSK utilizando filtro RRC com {m} portadoras')
plt.semilogy(SNRt, SERt)
plt.semilogy(SNRs, SERs, 'o:')
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.grid()
plt.show()

