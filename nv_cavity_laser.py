__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
import matplotlib.pyplot as plt

# constants
E = 7.7 * (10 ** -3)  # |-1> |1> distance
# gama = 2.25 * (10 ** 3)  # dissipation
# kappa = 5 * (10 ** 4)
gama = 0
kappa = 0
h = 1  # Plank
mu = 2.0028 * 13.99624 * (10 ** 0)  # electron ~ nv
B_z = 500.0  # magnetic field to split |1> |-1> spin states

w_p = 4.706 * (10 ** 5) - h * mu * B_z  # e->u transition frequency 10**14
w_m = w_p + h * E + h * mu * B_z  # e->g transition frequency
w_p_laser = w_p + 2 * 10 ** 4  # cavity frequency
w_c = w_m - 2 * 10 ** 4  # cavity frequency
delta = 0

omega_m = 100.0 * (10 ** 4)
omega_p = 100.0 * (10 ** 4)


def plot_populations(dt, ee, eg, eu, ge, gg, gu, t0, t1):
    x = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(x, ee, "black", label="ee")
    plt.plot(x, eg, "blue", label="e+")
    plt.plot(x, eu, "green", label="e-")
    plt.plot(x, ge, "gray", label="ge")
    plt.plot(x, gg, "red", label="g+")
    plt.plot(x, gu, "orange", label="g-")
    plt.show()


def integrate(dt, r, t0, t1):
    ee = linspace(t0, t1, (t1 - t0) / dt + 10)
    eg = linspace(t0, t1, (t1 - t0) / dt + 10)
    eu = linspace(t0, t1, (t1 - t0) / dt + 10)
    ge = linspace(t0, t1, (t1 - t0) / dt + 10)
    gg = linspace(t0, t1, (t1 - t0) / dt + 10)
    gu = linspace(t0, t1, (t1 - t0) / dt + 10)
    while r.successful() and r.t < t1:
        r.integrate(r.t + dt)
        ee[r.t / dt] = abs(r.y[0]) ** 2
        eg[r.t / dt] = abs(r.y[1]) ** 2
        eu[r.t / dt] = abs(r.y[2]) ** 2
        ge[r.t / dt] = abs(r.y[3]) ** 2
        gg[r.t / dt] = abs(r.y[4]) ** 2
        gu[r.t / dt] = abs(r.y[5]) ** 2
    return ee, eg, eu, ge, gg, gu


# Schrodinger equation's
def right_part(t, y):
    hamiltonian = array(
        [[1.5 * w_c - 1j * kappa, -.5 * omega_p * exp(-1j * t * (w_p - w_p_laser)), -.5 * omega_p * exp(-1j * t * (w_m - w_p_laser)),
          0., 0., 0.],
         [-.5 * omega_p * exp(1j * t * (w_p - w_p_laser)), 1.5 * w_c - 1j * kappa - w_p, 0.,
          -.5 * omega_m * exp(1j * t * (w_p - w_c)), 0., 0.],
         [-.5 * omega_p * exp(1j * t * (w_m - w_p_laser)), 0., 1.5 * w_c - 1j * kappa + delta - w_m - 1j * gama,
          -.5 * omega_m * exp(1j * t * (w_m - w_c)), 0., 0.],
         [0., -.5 * omega_m * exp(-1j * t * (w_p - w_c)), -.5 * omega_m * exp(-1j * t * (w_m - w_c)),
          .5 * w_c, -.5 * omega_p * exp(-1j * t * (w_p - w_p_laser)), -.5 * omega_p * exp(-1j * t * (w_m - w_p_laser))],
         [0., 0., 0.,
          -.5 * omega_p * exp(1j * t * (w_p - w_p_laser)), .5 * w_c - w_p, 0.],
         [0., 0., 0.,
          -.5 * omega_p * exp(1j * t * (w_m - w_p_laser)), 0., .5 * w_c + delta - w_m - 1j * gama]], dtype=complex128)
    hamiltonian *= -1j * h
    return dot(hamiltonian, y)


def create_integrator():
    r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex128)
    t0 = 0
    r.set_initial_value(psi_init, t0)
    return r, t0


def main():
    r, t0 = create_integrator()
    t1 = 5 * 10 ** -5
    dt = 10 ** -7
    ee, eg, eu, ge, gg, gu = integrate(dt, r, t0, t1)
    plot_populations(dt, ee, eg, eu, ge, gg, gu, t0, t1)


main()