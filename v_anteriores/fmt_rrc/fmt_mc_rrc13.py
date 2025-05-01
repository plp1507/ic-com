import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
from scipy.special import erfc

def rrc(alpha, Tb, k, l):

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

######## Características da SNR
npt = 10    # n. de pontos da simulação

SNRt = np.linspace(-8, 8, 1000) #SNR teórica
SNRs = SNRt[::1000//npt - 1]

SERt = (1/2)*erfc(np.sqrt(10**(SNRt/10)))

sigma2 = 10**(-SNRs/10)

SERs = np.zeros(len(sigma2))

######## Características da mensagem

n_blocos = 10**3   # n. de blocos
l_bloco = 20       # comprimento de cada bloco

m = 10             # n. de portadoras

l_filtro = 20      # intervalo de símbolos que o filtro dura

T = 1              # período de amostragem
Ts = l_bloco*T     # período do bloco de símbolo

######## Banco de filtros

alpha = l_bloco/m - 1

t, g_delay, f_rrc = rrc(alpha, Ts, l_bloco, l_filtro)
fk = np.arange(m)/(m*T)

banco_filtros = np.zeros([m, len(t)], dtype = 'complex')
for i in range(m):
    banco_filtros[i] = f_rrc*np.exp(2j*np.pi*fk[i]*t)
    if(i==0 or i==5):
        banco_filtros[i] *= np.sqrt(2)

'''
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Filtro RRC')
ax[0].plot(t, banco_filtros[0].real)
ax[0].grid()

ax[1].set_title('Espectro do filtro')
ax[1].plot(abs(np.fft.fftshift(np.fft.fft(banco_filtros[0]))))
ax[1].grid()

plt.show()
'''
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

for j, noise in enumerate(sigma2):
    w = np.sqrt(noise)*(np.random.randn(l_modl))
    sinal_rx = sinal_tx + w

    for i in range(m):
        z[i] = np.convolve(sinal_rx, np.conj(banco_filtros[i][::-1]))
        z_dn[i] = z[i][2*g_delay:-2*g_delay:l_bloco].real
        m_r[i] = np.sign(z_dn[i])
        plt.semilogy(10*np.log10(np.fft.fft(z[i])))
        #print(f'Erro na portadora {i}: {np.sum(m_r[i] != mensagem[i])/len(mensagem[i])}')
        #if(i != 0 and i != 5):
        SERs[j] += np.sum(m_r[i] != mensagem[i])/(len(mensagem[i]))

    SERs[j] /= m

    print(f'SER simulada: {SERs[j]}')
    print(f'SER teórica: {SERt[::1000//npt - 1][j]}')
    print('\n')


    plt.ylim(10**-1)    
    plt.grid()
    plt.show()


plt.title(f'Sistema FMT-BPSK com filtro RRC (alpha={round(alpha,2)}) com {m} portadoras')
plt.semilogy(SNRt, SERt, label = 'SER teórica')
plt.semilogy(SNRs, SERs, 'o:', label = 'SER simulada')
plt.xlabel('SNR (db)')
plt.ylabel('SER')
plt.grid()
plt.legend()
plt.show()
