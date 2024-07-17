#Python DFT implementation

import matplotlib.pyplot as plt
import numpy as np
import csv
from math import e, pi, cos, sin

#
#DFT:
#F(jw) = sum(k = 0 to N-1) of f(k)e^(-wjkT)
#

def DFT(signal, samplingTIME):
    DFT = np.zeros(len(signal), dtype = complex)
    
    for i in range(len(signal)-1):
        for k in range(len(signal)-1):
            exponent = -i*k*samplingTIME*(1j)
            DFT[i] += signal[k]*e**exponent
    
    ampltDFT = np.zeros(len(DFT), dtype = float)
    
    for i in range(len(DFT)):
        ampltDFT[i] = (np.real(DFT[i])**2) + (np.imag(DFT[i])**2)
        ampltDFT[i] = (ampltDFT[i])**0.5

    return ampltDFT

#placeholder signal
samplPERIOD = 0.002
signalTEST = np.zeros(2000, dtype = float)
time = np.arange(2000)
frequencies = np.arange(2000)

for i in range(2000):
    time[i] *= samplPERIOD
    signalTEST[i] = cos(35*i*samplPERIOD) + 3*cos(40*i*samplPERIOD) + 2

plt.plot(frequencies, DFT(signalTEST, samplPERIOD))
plt.show()
