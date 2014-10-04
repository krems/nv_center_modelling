from numpy import *
import numpy as np
B_z = 850.0
psi_init = array([0.0, 1.0, 0.0j], dtype=np.complex128)
D = 2877.0 * (10 ** 6)
E = 7.7 * (10 ** 6)
h = 4.135 * (10 ** -15);
# electron
mu = 2.0028 * 5.788 * (10 ** -9)
omega_field = (h * D + mu * B_z) / h

def B_0(t, A):
    return A * cos(t * 10 ** 9) 

def right_part(t, y):
    k1 = h * D
    k2 = h * E
    A = 750
    hamiltonian = array([
                  [k1 + mu * B_z, 
                   mu * B_0(t, A) * cos(omega_field * t), 
                   2 * k2], 
                  [mu * B_0(t, A) * cos(omega_field * t), 
                   0, 
                   mu * B_0(t, A) * cos(omega_field * t)],
                  [2 * k2, 
                   mu * B_0(t, A) * cos(omega_field * t), 
                   k1 - mu * B_z]
                  ], dtype=np.complex128)
    return dot(hamiltonian, y) * (-1j / h)

# Schrodinger equation    
from scipy.integrate import ode
r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
t0 = 0
r.set_initial_value(psi_init, t0)
t1 = 10 ** -8
dt = 10 ** -12
one = linspace(t0 + 0j, t1, (t1 - t0) / dt + 10)
two = linspace(t0 + 0j, t1, (t1 - t0) / dt + 10)
three = linspace(t0 + 0j, t1, (t1 - t0) / dt + 10)
while r.successful() and r.t < t1:
    r.integrate(r.t + dt)
    one[r.t / dt] = r.y[0]
    two[r.t / dt] = r.y[1]
    three[r.t / dt] = r.y[2]

import matplotlib.pyplot as plt
x = linspace(t0, t1, (t1 - t0) / dt + 10)
plt.plot(x, abs(one ** 2))
plt.plot(x, abs(two ** 2))
plt.plot(x, abs(three ** 2))
plt.show()