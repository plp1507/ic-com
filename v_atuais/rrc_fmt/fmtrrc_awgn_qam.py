import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
from scipy.special import erfc
from scipy.spatial.distance import cdist

def qam_constellation(M):
    range_ = int(np.sqrt(M))
    out = np.zeros([range_, range_], dtype = 'complex')
    for i in range(range_):
        for j in range(range_):
            out[i][j] = i + 1j*j
    out = np.reshape(out, M)
    out -= (np.sqrt(M)-1)*(.5 + .5j)
    e_total = np.sum(np.abs(out)**2)/M
    out /= np.sqrt(e_total)
    return out     

def decide(msg, constell):
    XA = np.column_stack((msg.real, msg.imag))
    XB = np.column_stack((constell.real, constell.imag))
    distcs = cdist(XA, XB, metric = 'euclidean')
    return constell[np.argmin(distcs, axis = 1)]

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


######## Características da mensagem

n_blocos = 10**4   # n. de blocos
l_bloco = 20       # comprimento de cada bloco

m = 10             # n. de portadoras
M = 4              #ordem da constelação

l_filtro = 20      # intervalo de símbolos que o filtro dura

T = 1              # período de amostragem
Ts = l_bloco*T     # período do bloco de símbolo

######## Características da SNR
npt = 10    # n. de pontos da simulação

SNRt = np.linspace(-2, 10, 1000) #SNR teórica
SNRs = SNRt[::1000//npt - 1]

argerfc = (3/(2*(M-1)))*(10**(SNRt/10))
p = (1-np.sqrt(1/M))*erfc(np.sqrt(argerfc))
SERt = 1 - ((1-p)**2)

sigma2 = 10**(-SNRs/10)

SERs = np.zeros(len(sigma2))

######## Banco de filtros
alpha = l_bloco/m - 1

t, g_delay, f_rrc = rrc(alpha, Ts, l_bloco, l_filtro)
fk = np.arange(m)/(m*T)

banco_filtros = np.zeros([m, len(t)], dtype = 'complex')
for i in range(m):
    banco_filtros[i] = f_rrc*np.exp(2j*np.pi*fk[i]*t)

######## Transmissão

qam_const = qam_constellation(M)
mensagem = np.random.choice(qam_const, [m, l_bloco*n_blocos])

l_modl = len(upfirdn(banco_filtros[0], mensagem[0], l_bloco))
m_modl = np.zeros([m, l_modl], dtype = 'complex')

for i in range(m):
    m_modl[i] = upfirdn(banco_filtros[i], mensagem[i], l_bloco)

sinal_tx = np.sum(m_modl, axis = 0)

######## Recepção
z = np.zeros([m, l_modl + len(f_rrc) - 1], dtype = 'complex')
z_dn = np.zeros([m, n_blocos*l_bloco], dtype = 'complex')
m_r = np.zeros([m, n_blocos*l_bloco], dtype = 'complex')

for j, noise in enumerate(sigma2):
    w = np.sqrt(noise/2)*(np.random.randn(l_modl)+1j*np.random.randn(l_modl))
    sinal_rx = sinal_tx + w

    for i in range(m):
        z[i] = np.convolve(sinal_rx, np.conj(banco_filtros[i][::-1]))
        z_dn[i] = z[i][2*g_delay:-2*g_delay:l_bloco]
        m_r[i] = decide(z_dn[i], qam_const)
        print(np.sum(m_r[i] != mensagem[i])/(l_bloco*n_blocos), i)
        SERs[j] += np.sum(m_r[i] != mensagem[i])/(l_bloco*n_blocos)

    SERs[j] /= m

    print(f'SER simulada: {SERs[j]}')
    print(f'SER teórica: {SERt[::1000//npt - 1][j]}')
    print('\n')


plt.title(f'Sistema FMT-QAM com filtro RRC (alpha={round(alpha,2)}) com {m} portadoras')
plt.semilogy(SNRt, SERt, label = 'SER teórica')
plt.semilogy(SNRs, SERs, 'o:', label = 'SER simulada')
plt.xlabel('SNR (db)')
plt.ylabel('SER')
plt.grid()
plt.legend()
plt.show()
