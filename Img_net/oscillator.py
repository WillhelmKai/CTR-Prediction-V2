#coding by Willhelm
#20190309
import numpy as np
import math
import os
import matplotlib.pyplot as plt

def oscillator(I):
    a1, a2, a3, a4 = 0.6, 0.6, -0.5, 0.5
    b1, b2, b3, b4 = -0.6, -0.6, -0.5, 0.5
    u,v,z,k= 0,0,0,50

    u_v = np.tanh(a1*u - a2*v + a3*z + a4*I)
    v_v = np.tanh(b3*z - b1*u - b2*v+b4*I)
    w = np.tanh(I)
    z_v = ( v_v - u_v )* np.exp(-k * (I**2))+w
    return z_v

x = np.arange(0,100)
x = x/100
y = oscillator(x)

plt.plot(x,y)
plt.show()