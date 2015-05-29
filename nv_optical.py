__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
import matplotlib.pyplot as plt

# constants
D = 2877.0 * (10 ** 6)  # g_u distance
E = 7.7 * (10 ** 6)  # |-1> |1> distance
gama = 2.25 * (10 ** 6)  # dissipation
# gama = 0
h = 1  # Plank
mu = 2.0028 * 5.788 * (10 ** -9)  # electron ~ nv
# todo: investigate real values for w_*
w_eu = 4.8367 * (10 ** 6)  # e->u transition frequency
w_eg = 4.8367 * (10 ** 6) + h * E  # e->g transition frequency
w_p = 4.8 * (10 ** 5)  # cavity frequency
delta = 4.706 * 10 ** 5  # detuning of cavity relatively to nv resonant transition

B_z = 850.0  # magnetic field to split |1> |-1> spin states
omega_s = 1.0 * (10 ** 4)
omega_p = 1.0 * (10 ** 4)


def plot_populations(dt, e, g, t0, t1, u):
    x = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(x, e)
    plt.plot(x, g)
    plt.plot(x, u)
    plt.show()


def integrate(dt, r, t0, t1):
    e = linspace(t0, t1, (t1 - t0) / dt + 10)
    g = linspace(t0, t1, (t1 - t0) / dt + 10)
    u = linspace(t0, t1, (t1 - t0) / dt + 10)
    while r.successful() and r.t < t1:
        r.integrate(r.t + dt)
        e[r.t / dt] = abs(r.y[0]) ** 2
        g[r.t / dt] = abs(r.y[1]) ** 2
        u[r.t / dt] = abs(r.y[2]) ** 2
    return e, g, u


# Schrodinger equation's
def right_part(t, y):
    hamiltonian = array(
        [[delta, omega_s, omega_p],
         [omega_s, -delta * 2., 0.0],
         [omega_p, 0.0, -delta * 2.]],
        dtype=complex128)
    hamiltonian *= -1j / h
    return dot(hamiltonian, y)


def create_integrator():
    r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([0.0, 0.0, 1.0], dtype=complex128)
    t0 = 0
    r.set_initial_value(psi_init, t0)
    return r, t0


def main():
    r, t0 = create_integrator()
    t1 = 10 ** -5
    dt = 10 ** -9
    e, g, u = integrate(dt, r, t0, t1)
    plot_populations(dt, e, g, t0, t1, u)


main()