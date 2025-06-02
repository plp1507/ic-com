import numpy as np
from matplotlib import pyplot as plt

#número de símbolos
n = 10

#comprimento do bloco multiportadora
l = 21

#geração dos símbolos aleatórios BPSK
mensagem = 2*np.round(np.random.rand(n)) - 1

#pulso protótipo
p = np.ones(l)

#normalização da energia do pulso
p /= np.linalg.norm(p)

#upsampling
sup = np.zeros(l*n)
sup[::l] = mensagem

#sinal após o filtro
x = np.convolve(sup, p)[:l*n]

fig, ax = plt.subplots(3, 1, figsize=(8,6))

ax[0].stem(sup)
ax[0].grid()
ax[1].plot(x)
ax[1].grid()

#modulação do sinal
xb = x
x = xb * np.exp(2j*np.pi*4*np.arange(l*n)/l)

ax[2].plot(x)
ax[2].grid()

plt.show()

#recepção do sinal
q = p[::-1]
xb = x*np.exp(-2j*np.pi*4*np.arange(n*l)/l)
z = np.convolve(xb, q)[:l*n]

#remoção do atraso de grupo e downsampling
z = z[2*int((l-1)/2)::l]

plt.plot(z)
plt.stem(mensagem)
plt.grid()
plt.show()
