import numpy as np
from matplotlib import pyplot as plt

#número de símbolos
n = 10

#comprimento do bloco multiportadora
l = 21

#pulso protótipo
p = np.ones(l)

#normalização da energia do pulso
p /= np.linalg.norm(p)

#geração dos símbolos aleatórios BPSK com 2 portadoras
mensagem = np.ones([2, n])
sup = np.zeros([2, l*n])
x = np.zeros([2, l*n], dtype = 'complex')

for i in range(2):
    mensagem[i] *= 2*np.round(np.random.rand(n))-1
    #upsampling
    sup[i][::l] = mensagem[i]
    #passagem pelo filtro
    x[i] = np.convolve(sup[i], p)[:l*n]

fig, ax = plt.subplots(3, 1, figsize=(8,6))

ax[0].stem(sup[0], linefmt = 'orange', label = 'portadora 1')
ax[0].stem(sup[1], label = 'portadora 2')
ax[0].grid()
ax[0].legend()
ax[1].plot(x[0], label = 'portadora 1')
ax[1].plot(x[1], label = 'portadora 2')
ax[1].grid()
ax[1].legend()

#modulação do sinal
xb = x
for i in range(2):
    x[i] = xb[i] * np.exp(2j*np.pi*(4+i)*np.arange(l*n)/l)

ax[2].plot(x[0], label = 'portadora 1')
ax[2].plot(x[1], label = 'portadora 2')
ax[2].grid()
ax[2].legend()
plt.show()

x = np.sum(x, axis=0)

#recepção do sinal
q = p[::-1]
z = np.zeros([2, l*n], dtype = 'complex')
z_d = np.zeros([2, n], dtype = 'complex')
for i in range(2):
    xb[i] = x*np.exp(-2j*np.pi*(4+i)*np.arange(n*l)/l)
    z[i] = np.convolve(xb[i], q)[:l*n]
    #remoção do atraso de grupo e downsampling
    z_d[i] = z[i][2*int((l-1)/2)::l]

fig, ax = plt.subplots(2, 1, figsize = (15,6))
ax[0].plot(z_d[0])
ax[0].stem(mensagem[0])
ax[0].grid()
ax[1].plot(z_d[1])
ax[1].stem(mensagem[1])
ax[1].grid()
plt.show()
