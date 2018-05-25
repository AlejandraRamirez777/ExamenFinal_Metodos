# Ejercicio5
# Resuelva el siguiente sistema acoplado de ecuaciones diferenciales 
# dx/dt = sigma * (y - x)
# dy/dt = rho * x - y -x*z
# dz/dt = -beta * z + x * y
# con sigma = 10, beta=2.67, rho=28.0,
# condiciones iniciales t=0, x=0.0, y=0.0, z=0.0, hasta t=5.0.
# Prepare dos graficas con la solucion: de x vs y (xy.png), x vs. z (xz.png) 

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def xt(x,y):
    return oo*(y-x)

def yt(x,y,z):
    return rr*(x-y-x)*z

def zt(x,y,z):
    return -1*bb*(z+x)*y

oo = 10
bb = 2.67
rr = 28.0


x = 0.0
t = 0.0
y = 0.0
z = 0.0
h = 0.001
dt = 0.01

xf = 0.0
yf = 0.0
zf = 0.0

xx = np.array([])
yy = np.array([])
zz = np.array([])

xx = np.append(xx,x)
yy = np.append(yy,x)
zz = np.append(zz,x)

while(t<5.0):
    
    xf = x + h*xt(x,y)
    yf = y + h*yt(x,y,z)
    zf =  z + h*zt(x,y,z)
    
    x = xf
    y = yf
    z = zf
    
    t += dt

plt.plot(y,x)
plt.title("x vs y")
plt.xlabel("y")
plt.ylabel("x")
plt.savefig("xy.png")
plt.clf()

plt.plot(z,x)
plt.title("x vs z")
plt.xlabel("y")
plt.ylabel("x")
plt.savefig("xz.png")
