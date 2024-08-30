import numpy as np
from math import e, pi
from matplotlib import pyplot as plt
from matplotlib import animation as anim

symbSize = 2
symbol = np.zeros(symbSize, int)
wordSize = 8
word = np.zeros([wordSize,symbSize], np.ndarray)

symbMarker = 0
wordMarker = 0

def generateRandomBit():
    return np.random.randint(2)

def constellationMap(symb):
    rotValue = ((symb[0] * 2) + symb[1])/(2**symbSize)
    rotAngle = rotValue*2*pi - pi/4
    return e**(1j*rotAngle)

while(1):
    #variável de estado da formação da palavra
    #false -> os N simbolos da palavra ainda nao foram gerados
    wordReady = False

    #fonte de bits
    bitStream = generateRandomBit()

    #conversão de serial para paralelo
    #bit -> simbolo -> palavra
    #entrada: bit; saída: palavra com N simbolos
    symbol[symbMarker] = bitStream
    symbMarker += 1
    if (symbMarker == symbSize):
        word[wordMarker] = symbol
        wordMarker += 1
        symbMarker = 0
    if (wordMarker == wordSize):
        wordReady = True
        wordMarker = 0

    #a partir daqui a palavra com o numero wordSize de simbolos já está formada
    if(wordReady):
        for i in range(wordSize):
           print(constellationMap(word[i]))
        wordReady = False
