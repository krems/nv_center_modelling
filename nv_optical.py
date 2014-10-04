__author__ = 'krm'

from numpy import *

B_z = 850.0
D = 2877.0 * (10 ** 6)
E = 7.7 * (10 ** 6)
# h = 4.135 * (10 ** -15);
h = 1
mu = 2.0028 * 5.788 * (10 ** -9)

w_eu = 4.8367 * (10 ** 5)
w_eg = 4.8367 * (10 ** 5) + h * E
# w_s = 4.3 * (10 ** 14)
w_p = 4.8 * (10 ** 5)
delta = w_eu - w_p
# delta = 0

def right_part(t, y):
    gama = 2.25 * (10 ** 6)
    omega_s = 45.0 * (10 ** 6)
    omega_p = 100.0 * (10 ** 6)
    hamiltonian = array([[delta - 1j * gama, omega_s, omega_p / 2.0 * sin(t * w_p)],
                         [omega_s, 0.0, 0.0],
                         [omega_p / 2.0 * sin(t * w_p), 0.0, 0.0]],
                        dtype=complex128)
    hamiltonian *= -1j / h
#     print hamiltonian
#     print y
#     print dot(hamiltonian, y)
    return dot(hamiltonian, y)

# Schrodinger equation
from scipy.integrate import ode
r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
psi_init = array([0.0, 1.0, 0.0], dtype=complex128)
t0 = 0
r.set_initial_value(psi_init, t0)
t1 = 10 ** -6
dt = 10 ** -12
e = linspace(t0, t1, (t1 - t0) / dt + 10)
g = linspace(t0, t1, (t1 - t0) / dt + 10)
u = linspace(t0, t1, (t1 - t0) / dt + 10)
while r.successful() and r.t < t1:
    r.integrate(r.t + dt)
    #print("%g %g %g %g" % (r.t, r.y[0], r.y[1], r.y[2]))
    e[r.t / dt] = abs(r.y[0]) ** 2
    g[r.t / dt] = abs(r.y[1]) ** 2
    u[r.t / dt] = abs(r.y[2]) ** 2
# print g, u, e

import matplotlib.pyplot as plt
x = linspace(t0, t1, (t1 - t0) / dt + 10)
plt.plot(x, e)
plt.plot(x, g)
plt.plot(x, u)
plt.show()