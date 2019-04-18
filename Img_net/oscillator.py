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

    u_v = math.tanh(a1*u + a2*v - a3*z + a4*I)
    v_v = math.tanh(b1*z - b2*u - b3*v + b4*I)
    w = math.tanh(I)
    z_v = ( v_v - u_v )* math.exp(-k*I*I)+ w
    return z_v, u_v, v_v


x = np.arange(-100,100)
x = x/100
N = 100
# for every single element
for i in range(0,len(x)):
    #each element in the time sequence T
    print('\r', "progress ",str(i/len(x)).ljust(10),end='')
    u, v, z = np.zeros(N),np.zeros(N),np.zeros(N)
    for time in range(0,N-1):
        z[time+1], u[time+1],v[time+1] = oscillator(x[i], u[time], v[time], z[time])
        plt.plot(x[i], z[time+1], 'c,')
plt.show()