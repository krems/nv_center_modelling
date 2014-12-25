__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
import matplotlib.pyplot as plt

# constants
D = 2877.0 * (10 ** 3)  # g_u distance
E = 7.7 * (10 ** 3)  # |-1> |1> distance
gama = 2.25 * (10 ** 6)  # dissipation
# gama = 0
h = 1  # Plank
mu = 2.0028 * 5.788 * (10 ** 3)  # electron ~ nv
# todo: investigate real values for w_*
w_p = 4.8367 * (10 ** 6)  # e->u transition frequency
w_m = w_p + h * E  # e->g transition frequency
w_z = w_m + h * D
w_f = 4.8 * (10 ** 6)  # cavity frequency
delta = w_p - w_f  # detuning of cavity relatively to nv resonant transition

B_z = 500.0  # magnetic field to split |1> |-1> spin states
omega_s = 100.0 * (10 ** 6)
omega_p = 100.0 * (10 ** 6)
omega_z = 30.0 * (10 ** 5)


def plot_populations(dt, e, g, t0, t1, u, z):
    x = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(x, e, "green", label="+")
    plt.plot(x, g, "blue", label="-")
    plt.plot(x, u, "black", label="0")
    plt.plot(x, z, "red", label="e")
    plt.show()


def integrate(dt, r, t0, t1):
    e = linspace(t0, t1, (t1 - t0) / dt + 10)
    g = linspace(t0, t1, (t1 - t0) / dt + 10)
    u = linspace(t0, t1, (t1 - t0) / dt + 10)
    z = linspace(t0, t1, (t1 - t0) / dt + 10)
    while r.successful() and r.t < t1:
        r.integrate(r.t + dt)
        e[r.t / dt] = abs(r.y[0]) ** 2
        g[r.t / dt] = abs(r.y[1]) ** 2
        u[r.t / dt] = abs(r.y[2]) ** 2
        z[r.t / dt] = abs(r.y[3]) ** 2
    return e, g, u, z


# Schrodinger equation's
def right_part(t, y):
    # a = 700
    hamiltonian = array(
        [[-w_p, -h * E / 2., -h * D / 2. - h * mu * B_z, -omega_p / 2. * exp(1j * t * w_f)],
         [-h * E / 2., -w_m, -h * D / 2. - h * mu * B_z, -omega_s / 2. * exp(1j * t * w_f)],
         [-h * D / 2. - h * mu * B_z, -h * D / 2. - h * mu * B_z, -((w_p + w_m) / 2 - h * D), -omega_z / 2. * exp(1j * t * w_f)],
         [-omega_p / 2. * exp(-1j * t * w_f), -omega_s / 2. * exp(-1j * t * w_f), -omega_z / 2. * exp(-1j * t * w_f), delta - 1j * gama]],
        dtype=complex128)
    hamiltonian *= -1j / h
    return dot(hamiltonian, y)


def create_integrator():
    r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([1.0, 0.0, 0.0, 0.0], dtype=complex128)
    t0 = 0
    r.set_initial_value(psi_init, t0)
    return r, t0


def main():
    r, t0 = create_integrator()
    t1 = 10 ** -6
    dt = 10 ** -11
    e, g, u, z = integrate(dt, r, t0, t1)
    plot_populations(dt, e, g, t0, t1, u, z)


main()