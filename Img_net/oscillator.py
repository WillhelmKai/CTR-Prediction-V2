#coding by Willhelm
#20190309
import numpy as np
import math
import os
import matplotlib.pyplot as plt

def oscillator(I, u, v, z):
    a1, a2, a3, a4 = 0.6, 0.6, -0.5, 0.5
    b1, b2, b3, b4 = -0.6, -0.6, -0.5, 0.5
    k = 50

    u_v = np.tanh(a1*u - a2*v + a3*z + a4*I)
    v_v = np.tanh(b3*z - b1*u - b2*v+b4*I)
    w = np.tanh(I)
    z_v = ( v_v - u_v )* np.exp(-k * (I**2))+w
    return z_v, u_v, v_v

u, v, z= 0, 0, 0
x = np.arange(0,100)
x = x/100
y = []
for i in range(0,len(x)):
    z_temp,u_temp,v_temp= oscillator(x[i], u, v, z)
    u,v,z = u_temp,v_temp,z_temp
    y.append(z)
y = np.array(y)

plt.plot(x,y)
plt.show()