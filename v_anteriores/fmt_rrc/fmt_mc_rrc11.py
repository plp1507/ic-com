'''
Geração de símbolos BPSK em sistema FMT utilizando filtro RRC
Autor: Pedro L.
Data: 27/03/2025
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
### configuração do filtro RRC

alpha = 1    ####deve ser maior que 0.2
k = 10        #fator de oversampling
Fs = 20     #frequência de amostragem (amostras/s)
Ts = k/Fs      #período de símbolo (s)
l = 50          #comprimento do filtro (segundos)

x = np.linspace(-l/2, l/2, l*Fs+1)/Ts
filtro_rrc = (1/Ts)*(np.sin(np.pi*x*(1-alpha)) + (4*alpha*x)*np.cos(np.pi*x*(1+alpha)))

filtro_rrc[x==0] = (1/Ts)*(1+alpha*(4/np.pi - 1))
filtro_rrc[abs(x) == 0.25*Ts/alpha] = (alpha/(Ts*np.sqrt(2)))*((1+2/np.pi)*np.sin(0.25*np.pi/alpha)+ (1-2/np.pi)*np.cos(0.25*np.pi/alpha))
filtro_rrc[x!=0] /= np.pi*x[x!=0]*(1-(4*alpha*x[x!=0])**2)

filtro_rrc /= np.sqrt(np.sum(filtro_rrc**2))

### configuração da mensagem
n = 10**5    #número de símbolos
m = 4     #número de portadoras

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
    m_modl[i] = m_conv[i]*np.exp(-2j*np.pi*(i + m)*np.arange(0, n*k + Fs*l)/m)
    plt.plot(10*np.log10(abs(np.fft.fft(m_modl[i]))))

plt.grid()
plt.show()

sinal_tx = np.sum(m_modl, axis=0)

#####recepção
#demodulação, passagem por filtro casado e downsampling
m_demod = np.zeros([m, n*k + Fs*l], dtype= 'complex')
m_casad = np.zeros([m, n*k + 2*Fs*l], dtype = 'complex')
m_dwnsp = np.zeros([m, n])

#SNR
npt = 10

SNRt = np.linspace(0, 10, 1000)
SNRs = SNRt[::1000//npt - 1]
sigma2 = 10**(-SNRs/10)

SERt = (1/2)*erfc(np.sqrt(10**(SNRt/10)))
SERs = np.zeros(npt+1)

for j, noise in enumerate(sigma2):
    w = np.sqrt(noise)*(np.random.randn(n*k + l*Fs))
    sinal_rx = sinal_tx + w
    
    for i in range(m):
        m_demod[i] = sinal_rx*np.exp(2j*np.pi*(i + m)*np.arange(0, n*k + Fs*l)/m)
        m_casad[i] = np.convolve(m_demod[i], filtro_rrc)
        m_dwnsp[i] = m_casad[i][l*Fs:l*Fs + n*k:k].real
        #plt.plot(abs(np.fft.fftshift(np.fft.fft(m_casad[i]))))

    #plt.grid()
    #plt.show()

    m_reconstruida = np.sign(np.reshape(m_dwnsp, -1))
    SERs[j] = np.sum(m_reconstruida != mensagem)/(m*n)

    print('\n')
    print(f'SNR simulada: {round(10*np.log10(np.mean(abs(m_reconstruida)**2)/np.mean(abs(w)**2)),2)} dB')
    print(f'SNR teórica:  {round(SNRs[j], 2)} dB\n')
    print(f'SER simulada: {round(SERs[j], 6)}')
    print(f'SER teórica:  {round((1/2)*erfc(np.sqrt(10**(SNRs[j]/10))),6)}\n')

plt.semilogy(SNRt, SERt, label = 'teorico')
plt.semilogy(SNRs, SERs, 'o:', label = 'simulado')
plt.grid()

plt.xlabel('SNR (dB)')
plt.ylabel('SER')

plt.title(f'Sistema FMT-BPSK com {m} portadoras')

plt.show()
