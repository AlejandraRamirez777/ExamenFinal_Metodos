#Ejercicio4
# 'y' es una senal en funcion del tiempo 't' con las unidades descritas en el codigo.
# a. Grafique la senal en funcion del tiempo en la figura 'senal.png' ('y' vs. 't')
# b. Calule la transformada de Fourier (sin utilizar funciones de fast fourier transform) y
# grafique la norma de la transformada en funcion de la frecuencia (figura 'fourier.png')
# c. Lleve a cero los coeficientes de Fourier con frecuencias mayores que 10000 Hz y calcule 
# la transformada inversa para graficar la nueva senal (figura 'filtro.png')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = 512 # number of point in the whole interval
f = 200.0 #  frequency in Hz
dt = 1 / (f * 128 ) #128 samples per frequency
t = np.linspace( 0, (n-1)*dt, n) 
y = np.sin(2 * np.pi * f * t) + np.cos(2 * np.pi * f * t * t)
noise = 1.4*(np.random.rand(n)+0.7)
y  =  y + noise

plt.plot(t,y)
plt.title("Senal en funcion de t")
plt.xlabel("t")
plt.ylabel("y")
plt.savefig("senal.png")
plt.clf()

fff = np.zeros(n)

for k in range(n):
    g = 0
    for nn in range(n-1):
        w = nn/float(n)*dt
        m = np.exp(-1j*2.0*w*k)
        g += y[k]*m
    fff[k] = g
    
plt.plot(t,fff)
plt.title("Transformada")
plt.xlabel("f")
plt.ylabel("ft")
plt.savefig("fourier.png")
plt.clf()

for i in fff:
    if(i>1000):
        i=0

fffi = np.zeros(n)
for k in range(n):
    g = 0
    for nn in range(n-1):
        w = nn/float(n)*dt
        m = np.exp(1j*2.0*w*k)
        g += y[k]*m
    fffi[k] = g
    
plt.plot(t,fffi)
plt.title("Transformada inversa vs t")
plt.xlabel("t")
#plt.ylabel("y")
plt.savefig("filtro.png")
