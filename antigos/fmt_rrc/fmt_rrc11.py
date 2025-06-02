import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn

def rrc(alpha, Tb, k, l):

    ###
    # alpha alpha
    # Tb    Tb
    # L     k
    # span  l

    Ts = Tb/k

    t = np.arange(-l*Tb/2, l*Tb/2 + Ts, Ts)

    with np.errstate(divide='ignore', invalid='ignore'):
        filtro_rrc = np.divide(np.sin(np.pi*t*(1-alpha)/Tb) + 
                (4*alpha*t/Tb)*np.cos(np.pi*t*(1+alpha)/Tb),
                        (np.pi*t/Tb)*(1 - (4*alpha*t/Tb)**2))

    filtro_rrc /= Tb

    filtro_rrc[np.argwhere(np.isnan(filtro_rrc))] = (1/Tb)*(1 - alpha + 4*alpha/np.pi)
    filtro_rrc[np.argwhere(np.isinf(filtro_rrc))] = ((alpha/(Tb*np.sqrt(2))) * ((1 + (2/np.pi))*np.sin(np.pi/(4*alpha)) + (1 - (2/np.pi))*np.cos(np.pi/(4*alpha))))

    atraso = l*Tb//2

    filtro_rrc /= np.sqrt(np.sum(filtro_rrc**2))

    return t, atraso, filtro_rrc

######## Características da mensagem

n_blocos = 10**3   # n. de blocos
l_bloco = 13    # comprimento de cada bloco

m = 10           # n. de portadoras

l_filtro = 20   # intervalo de símbolos que o filtro dura

T = 1           # período de amostragem
Ts = l_bloco*T  # período do bloco de símbolo

######## Banco de filtros

alpha = l_bloco/m - 1

t, g_delay, f_rrc = rrc(alpha, Ts, l_bloco, l_filtro)
fk = np.arange(m)/(m*T)

banco_filtros = np.zeros([m, len(t)], dtype = 'complex')
for i in range(m):
    banco_filtros[i] = f_rrc*np.exp(2j*np.pi*fk[i]*t)

fig, ax = plt.subplots(1, 2)
ax[0].set_title('Filtro RRC')
ax[0].plot(t, banco_filtros[0].real)
ax[0].grid()

ax[1].set_title('Espectro do filtro')
ax[1].plot(abs(np.fft.fftshift(np.fft.fft(banco_filtros[0]))))
ax[1].grid()

plt.show()

######## Transmissão

mensagem = 2*np.round(np.random.rand(m, l_bloco*n_blocos)) - 1

l_modl = len(upfirdn(banco_filtros[0], mensagem[0], l_bloco))
m_modl = np.zeros([m, l_modl], dtype = 'complex')

for i in range(m):
    m_modl[i] = upfirdn(banco_filtros[i], mensagem[i], l_bloco)

sinal_tx = np.sum(m_modl, axis = 0)

######## Recepção

z = np.zeros([m, l_modl + len(f_rrc) - 1], dtype = 'complex')
z_dn = np.zeros([m, n_blocos*l_bloco])
m_r = np.zeros([m, n_blocos*l_bloco])

for i in range(m):
    z[i] = np.convolve(sinal_tx, np.conj(banco_filtros[i][::-1]))
    z_dn[i] = z[i][2*g_delay:-2*g_delay:l_bloco].real
    m_r[i] = np.sign(z_dn[i])
    print(f'Erro por portadora: {np.sum(m_r[i] != mensagem[i])/len(mensagem[i])}\n')

