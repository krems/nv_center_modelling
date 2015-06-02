# -*- coding: utf-8 -*-
__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
import matplotlib.pyplot as plt

# constants
E = 7.7 * (10 ** -3)  # |-1> |1> distance [shifted]
D = 2877.0 * (10 ** -3)  # [shifted]
h = 1.  # Plank
# normalization 10 ** -9
mu = 2.0028 * 13.99624 * (10 ** 0)  # electron ~ nv [shifted]
B_z = 0.0  # magnetic field to split |1> |-1> spin states

w_p = 4.706 * (10 ** 5) - mu * B_z  # e<->u transition frequency [shifted]
w_m = w_p + E + mu * B_z  # e<->g transition frequency [shifted]
w_l = w_p  # + 2 * 10 ** 4  # cavity frequency [shifted]
w_c = w_m  # - 2 * 10 ** 4  # cavity frequency [shifted]
delta = 0.  # [shifted]
# gama = 5. * (10 ** -1)  # dissipation from NV [shifted]
gama = 0
# kappa = w_c / (2.4 * 10 ** 6)  # dissipation from cavity [shifted]
kappa = 0

omega_l_m = 1.0 * (10 ** 4)  # [shifted]
omega_l_p = 1.0 * (10 ** 4)  # [shifted]
omega_c_m = 0.0 * (10 ** 4)  # [shifted]
omega_c_p = 0.0 * (10 ** 4)  # [shifted]

cav_g = mat(array([0.0, 1.0]), dtype=complex128)
cav_e = mat(array([1.0, 0.0]), dtype=complex128)

nv_e = mat(array([1.0, 0.0, 0.0]), dtype=complex128)
nv_p = mat(array([0.0, 1.0, 0.0]), dtype=complex128)
nv_m = mat(array([0.0, 0.0, 1.0]), dtype=complex128)

ee = mat(kron(cav_e, nv_e))
ep = mat(kron(cav_e, nv_p))
em = mat(kron(cav_e, nv_m))
ge = mat(kron(cav_g, nv_e))
gp = mat(kron(cav_g, nv_p))
gm = mat(kron(cav_g, nv_m))

E_two = matrix([[1.0, 0.0],
                [0.0, 1.0]], dtype=complex128)
E_three = matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=complex128)

b = kron(E_two, nv_m.T.dot(nv_e))
d = kron(E_two, nv_p.T.dot(nv_e))
a = kron(cav_g.T.dot(cav_e), E_three)


def plot_populations(dt, ee, eg, eu, ge, gg, gu, t0, t1):
    # t = linspace(t0 * 1000, t1 * 1000, (t1 - t0) / dt + 10)
    t = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(t, ee, "red", label="|e,e>", linestyle='--')
    plt.plot(t, eg, "blue", label="|e,+>", linewidth=2.5, linestyle=':')
    plt.plot(t, eu, "green", label="|e,->", linewidth=2.5, linestyle='-.')
    plt.plot(t, ge, "orange", label="|g,e>", linewidth=2.5, linestyle='--')
    plt.plot(t, gg, "black", label="|g,+>", linestyle=':')
    plt.plot(t, gu, "gray", label="|g,->", linestyle='-.')
    plt.xlabel(u't, нс')
    plt.ylabel(u'Заселенности')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def commutator(a, b):
    return a.dot(b) - b.dot(a)


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
    ham_c = -h * w_c * a.T.dot(a)
    ham_nv = -h * (delta * b.T.dot(b) + w_p * d.dot(d.T) + w_m * b.dot(b.T))
    ham_nv_c = -h * omega_c_m * (b.T.dot(a) + b.dot(a.T)) - h * omega_c_p * (d.T.dot(a) + d.dot(a.T))
    ham_nv_laser = -h * omega_l_m * (b.T + b) - h * omega_l_p * (d.T + d)
    dissipation = -1j * gama * b.T.dot(b) - 1j * kappa * a.dot(a.T)
    hamiltonian = ham_c + ham_nv + ham_nv_c + ham_nv_laser + dissipation
    return dot(hamiltonian, y) * (-1j / h)


def create_integrator():
    r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex128)
    t0 = 0.
    r.set_initial_value(psi_init, t0)
    return r, t0


def main():
    r, t0 = create_integrator()
    t1 = 2. * 10 ** -2
    dt = 3. * 10 ** -5
    ee, eg, eu, ge, gg, gu = integrate(dt, r, t0, t1)
    plot_populations(dt, ee, eg, eu, ge, gg, gu, t0, t1)


main()
