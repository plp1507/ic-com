'''
Geração de símbolos BPSK em sistema FMT utilizando filtro RRC
Autor: Pedro L.
Data: 27/03/2025
'''

import numpy as np
from matplotlib import pyplot as plt

### configuração do filtro RRC

alpha = 1    ####deve ser maior que 0.2
k = 10        #fator de oversampling
Fs = 200     #frequência de amostragem (amostras/s)
Ts = k/Fs      #período de símbolo (s)
l = 20          #comprimento do filtro (segundos)

x = np.linspace(-l/2, l/2, l*Fs+1)/Ts
filtro_rrc = (1/Ts)*(np.sin(np.pi*x*(1-alpha)) + (4*alpha*x)*np.cos(np.pi*x*(1+alpha)))

filtro_rrc[x==0] = (1/Ts)*(1+alpha*(4/np.pi - 1))
filtro_rrc[abs(x) == 0.25*Ts/alpha] = (alpha/(Ts*np.sqrt(2)))*((1+2/np.pi)*np.sin(0.25*np.pi/alpha)+ (1-2/np.pi)*np.cos(0.25*np.pi/alpha))
filtro_rrc[x!=0] /= np.pi*x[x!=0]*(1-(4*alpha*x[x!=0])**2)

filtro_rrc /= np.sqrt(np.sum(filtro_rrc**2))

plt.plot(abs(np.fft.fftshift(np.fft.fft(filtro_rrc))))
plt.grid()
plt.show()

### configuração da mensagem
n = 10    #número de símbolos
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
    m_modl[i] = m_conv[i]*np.exp(-2j*np.pi*(i)*np.arange(0, n*k + Fs*l)/m)


sinal_tx = np.sum(m_modl, axis=0)

plt.plot(abs(np.fft.fftshift(np.fft.fft(sinal_tx))))
plt.grid()
plt.show()

#####recepção
###demodulação, passagem por filtro casado e downsampling
m_demod = np.zeros([m, n*k + Fs*l], dtype= 'complex')
m_casad = np.zeros([m, n*k + 2*Fs*l], dtype = 'complex')
m_dwnsp = np.zeros([m, n])

noise = 0.000001

for i in range(m):

    sinal_tx += np.sqrt(noise/2)*(np.random.randn(n*k + l*Fs))

    m_demod[i] = sinal_tx*np.exp(2j*np.pi*(i+m)*np.arange(0, n*k + Fs*l)/m)
    m_casad[i] = np.convolve(m_demod[i], filtro_rrc)
    m_dwnsp[i] = m_casad[i][l*Fs:l*Fs + n*k:k].real


m_reconstruida = np.reshape(m_dwnsp, -1)

fig, ax = plt.subplots(2, 1)

for i in range(m):
    ax[0].plot(m_paralelo[i])
    ax[1].plot(m_dwnsp[i], ':')

ax[0].grid()
ax[1].grid()
plt.show()
