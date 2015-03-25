__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
import matplotlib.pyplot as plt

# constants
E = 7.7 * (10 ** -3)  # |-1> |1> distance
gama = 2.25 * (10 ** 4)  # dissipation
# gama = 0
h = 1  # Plank
mu = 2.0028 * 13.99624 * (10 ** 0)  # electron ~ nv
B_z = 500.0  # magnetic field to split |1> |-1> spin states

w_p = 4.706 * (10 ** 5) - h * mu * B_z  # e->u transition frequency 10**14
w_m = w_p + h * E + h * mu * B_z  # e->g transition frequency
w_p_laser = w_p  # cavity frequency
w_m_laser = w_m  # cavity frequency
delta = 0

omega_m = 100.0 * (10 ** 4)
omega_p = 100.0 * (10 ** 4)


def plot_populations(dt, e, g, u, t0, t1):
    x = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(x, e, "green", label="e")
    plt.plot(x, g, "blue", label="+")
    plt.plot(x, u, "black", label="-")
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
        [[2 * delta - 1j * gama / h, -omega_p * exp(1j * t * (w_p - w_p_laser)) -omega_m * exp(1j * t * (w_p - w_m_laser)), -omega_m * exp(1j * t * (w_m - w_m_laser))-omega_p * exp(1j * t * (w_m - w_p_laser))],
         [-omega_p * exp(- 1j * t * (w_p - w_p_laser))-omega_m * exp(- 1j * t * (w_p - w_m_laser)), -2 * w_p, 0],
         [-omega_m * exp(- 1j * t * (w_m - w_m_laser))-omega_p * exp(- 1j * t * (w_m - w_p_laser)), 0, -2 * w_m]],
        dtype=complex128)
    hamiltonian *= -1j / 2
    return dot(hamiltonian, y)


def create_integrator():
    r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([0.0, 0.0, 1.0], dtype=complex128)
    t0 = 0
    r.set_initial_value(psi_init, t0)
    return r, t0


def main():
    r, t0 = create_integrator()
    t1 = 2 * 10 ** -5
    dt = 10 ** -8
    e, g, u = integrate(dt, r, t0, t1)
    plot_populations(dt, e, g, u, t0, t1)


main()