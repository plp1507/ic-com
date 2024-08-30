import numpy as np
from matplotlib import pyplot as plt
from math import e, pi, cos, floor

'''
xn = (1/N)*sum(k = 0, N-1): Xk*e**(j2pi(k/N)n)
'''

lenSig = 1000
freqDistr1 = np.zeros(lenSig, dtype = "complex_")
freqDistr2 = np.zeros(lenSig, dtype = "complex_")
freqDistr3 = np.zeros(lenSig, dtype = "complex_")
freqDistr4 = np.zeros(lenSig, dtype = "complex_")

finalSignal = np.zeros(lenSig, dtype = "complex_")

def IDFT(signal):
    lenSign = len(signal)
    timeDistr = np.zeros(lenSign, dtype = "complex_")
    timeAmplt = np.zeros(floor(lenSign/2), dtype = float)
    
    multConst = 1/lenSign

    for i in range(floor(lenSign/2)):
        for j in range(lenSign):
            expConst = 2j*i*j*pi/lenSign
            complxExp = e**(expConst)
            timeDistr[i] += complxExp*signal[j]
        timeDistr[i] *= 1/lenSign
        timeAmplt[i] = timeDistr[i].real**2 + timeDistr[i].imag**2
        timeAmplt[i] = timeAmplt[i]**0.5
    
    return timeAmplt

window = 100
df = 1
for i in range(window):
    freqDistr1[i] = cos(0.5*i/pi)
    freqDistr2[i] = cos(1.0*i/pi)
    freqDistr3[i] = cos(1.5*i/pi)
    freqDistr4[i] = cos(2.0*i/pi)

for i in range(lenSig):
    finalSignal[i] = freqDistr1[i] + freqDistr2[i]

time1 = IDFT(freqDistr1)
time2 = IDFT(freqDistr2)
time3 = IDFT(freqDistr3)
time4 = IDFT(freqDistr4)
time5 = IDFT(finalSignal)
plt.plot(time1)
plt.plot(time2)
plt.plot(time3)
plt.plot(time4)
#plt.plot(time5)
plt.show()
