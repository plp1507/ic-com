import numpy as np
from math import erfc
from matplotlib import pyplot as plt

len = 100

EbN0 = np.linspace(-2, 10, len)
EbN0norm = (10**(EbN0/10))
BER = np.zeros(len)

for i in range(len):
    BER[i] = erfc(np.sqrt(EbN0norm[i]))/2

plt.plot(EbN0, BER)
plt.yscale('log')
plt.show()
