import numpy as np
from math import e, pi, ceil, floor
from matplotlib import pyplot as plt

#Parâmetros
N = 10000
M = 1
Ncp = 10

E0 = 100

N0 = 3

def rect_constellation(m):
    out = np.zeros([range1, range2], dtype= "complex_")
    range1 = floor(np.sqrt(m))
    range2 = ceil(np.sqrt(m))
    for i in range(range1):
        for j in range(range2):
            out[i][j] = 2* i + 2j* j

    out = out.reshape(m)
    out -= (np.sqrt(m)-1)*(1+1j)
    return out

constel = rect_constellation(8)
plt.scatter(constel.real, constel.imag)
plt.show()
#Constelação
#constell = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])


#Escolha aleatória dos símbolos
#Xl = np.random.choice(constell, N*M)

#monta a matriz NxM (N símbolos de tamanho M)
#X = np.reshape(Xl, (N,M), order = 'F')

#faz a IDFT
#x = np.fft.ifft(X, axis = 0, norm = 'ortho')

#adiciona o prefixo cíclico de tamanho Ncp
#xcp = np.vstack((x[N-Ncp:], x))

#coloca os dados em série
#xs = xcp.reshape(-1, order = 'F')


#passagem pelo canal
#v = np.random.randn((N+Ncp)*M)

    
#caminho reverso

#coloca os dados em formato de matriz NxM (paralelo)
#xcpR = np.reshape(xs, (N+Ncp,M), order = 'F')

#retira o prefixo cíclico
#xR = np.delete(xcpR, np.arange(Ncp), 0)

#faz a dft
#XR = np.fft.fft(xR, axis = 0, norm = 'ortho')

#coloca os dados em série
#XlR = XR.reshape(-1, order = 'F')

#inf_recebida = np.zeros(N*M, dtype = "complex_")

#bit_errado = 0

#detecçao pela distancia euclidiana

#print((bit_errado/N*M)*100)

#plt.scatter(Xl.real, Xl.imag, s=50)
#plt.scatter(XlR.real, XlR.imag, s=15)
#plt.show()
