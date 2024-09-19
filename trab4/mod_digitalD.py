import numpy as np
from matplotlib import pyplot as plt
from math import erfc, floor, ceil

constell = np.arange(0, 8)

teorico = np.linspace(0, 15, 100)

SER = np.zeros(100)
k = 0

dist = np.zeros(len(constell))

def constellation(m):
    rang = int(np.sqrt(m))

    out = np.zeros([range_, range_], dtype = "complex_")
    for i in range(range_):
        for k in range(range_):
            out[i][k] = 2*i + 2j*k

    out = out.reshape(m)
    out -= (np.sqrt(m)-1)*(1+1j)
    return out

def detec(msg):
    msg_range1 = complex(ceil(msg.real) + 1j*ceil(msg.imag))
    msg_range2 = complex(floor(msg.real) + 1j*ceil(msg.imag))
    msg_range3 = complex(ceil(msg.real) + 1j*floor(msg.imag))
    msg_range4 = complex(floor(msg.real) + 1j*floor(msg.imag))
    subq = np.array([msg_range1, msg_range2, msg_range3, msg_range4])
    
    dists = np.linalg.norm(subq)

    return subq[np.argmin(dists)]

for valores in teorico:
    N0 = (10**(-valores/10))
    N = 10000000

    mensagem = np.random.choice(constell, N)

    #passagem pelo canal

    noise = (N0/2)*np.random.randn(N)

    recebido = mensagem + noise
    
    detecd = np.zeros(N, dtype = "complex_")
    for l in range(N):
        detecd[l] = detec(recebido[l])
    
    taxaErro = np.sum(detecd != mensagem)/N

    SER[k] = taxaErro
    k+=1

print(SER)
