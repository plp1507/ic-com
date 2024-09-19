import numpy as np
from math import erfc, log
from matplotlib import pyplot as plt

len = 100
EbN0 = np.linspace(-2, 15, len)
No = (10**(-EbN0/20))
N0 = (10**(EbN0/10))

SER = np.zeros(len)
constell = np.array ([-1, 1])/np.sqrt(2)

j = 0

for valores in No:
    N = 10000000

    mensagem = np.random.choice(constell, N)

    #passagem pelo canal

    noise = (valores/2)*np.random.randn(N)

    recebido = mensagem + noise

    taxaErro = np.sum(np.sign(recebido) != np.sign(mensagem))/N
    SER[j] = taxaErro
    j += 1
   
BER = np.zeros(len)
for i in range(len):
    BER[i] = erfc(np.sqrt(N0[i]))/2

plt.plot(EbN0, BER)
plt.plot(EbN0, SER)
plt.yscale('log')
plt.show()
