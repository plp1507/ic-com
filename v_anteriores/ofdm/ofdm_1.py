import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

#parâmetros do sistema
N = 100000
M = 1
Ncp = 0

mPAM = 1024

EbN0dB= np.linspace(-2, 15, 100)

#EbN0dB = -2

#EbNodB = 10log(Eb/N0)
#EbN0 = 10**(EbN0dB/10)
#1/N0 = 10**(EbN0dB/10)
N0 = 10**(-EbN0dB/10)

ber = np.zeros(len(EbN0dB))
k=0

#função para gerar a constelação como vetor de tamanho mPAM
def constellation(m):
    range_ = int(np.sqrt(m))
    out = np.zeros([range_,range_], dtype = "complex_")
    for i in range(range_):
        for j in range(range_):
            out[i][j] = 1*i + 1j*j
    
    out = np.reshape(out, m)

    #p/ centrar a const. em 0
    out -= (np.sqrt(m)-1)*(0.5 +0.5j)

    #normalização pela energia média da constelação
    energTotal = np.sum(np.linalg.norm(out))
    out /= energTotal
    return out

constel = constellation(mPAM)

plt.scatter(constel.real, constel.imag)
plt.show()

def detec(msgR):
    for i in range(mPAM):
        XA = np.column_stack((msgR.real, msgR.imag))
        XB = np.column_stack((constel.real, constel.imag))
        distcs = cdist(XA, XB, metric = 'euclidean')
    return constel[np.argmin(distcs)]

for Noise in N0:
    #escolha aleatória de pontos da constelação, mundança serie/paralelo
    X = np.random.choice(constel, N*M)
    Xl = np.reshape(X, [N, M], order = 'F')

    #aplicação da IDFT, adição do prefx. cíclico e conversão paralelo/serie
    x = np.fft.ifft(Xl, norm = 'ortho')
    xcp = np.vstack((x[N-Ncp:], x))
    xcp = np.reshape(xcp,((N+Ncp)*M), order = 'F')

    #passagem pelo canal AWGN
    v = Noise*(np.random.randn((N+Ncp)*M) + 1j*np.random.randn((N+Ncp)*M))
    xcpR = xcp + v

    #conversão serie/paralelo, remoção do prefx. cíclico e aplicação da DFT
    xcpR = np.reshape(xcpR, [N+Ncp, M], order = 'F')
    xR = np.delete(xcpR, range(Ncp), axis=0)
    XlR = np.fft.fft(xR, norm = 'ortho')

    XR = np.reshape(XlR, N*M, order = 'F')

    XRd = np.zeros(N*M, dtype = "complex_")
    for i in range(N*M):
        XRd[i] = detec(XR[i])

    ber[k]  = np.sum(XRd != X)/(N*M)
    k+=1

#plt.plot(EbN0dB, ber)
#plt.yscale('log')
#plt.show()
