# -*- coding: utf-8 -*-
__author__ = 'valerii ovchinnikov'

from numpy import *
from odeintw import odeintw
import matplotlib.pyplot as plt

# constants
E = 7.7 * (10 ** -3)  # |-1> |1> distance
h = 1.  # Plank
mu = 2.0028 * 13.99624 * (10 ** 0)  # electron ~ nv
B_z = 0.0  # magnetic field to split |1> |-1> spin states

w_p = 4.706 * (10 ** 3) - mu * B_z  # e<->u transition frequency 10**14
w_m = w_p + E + mu * B_z  # e<->g transition frequency
w_c = w_m  # cavity frequency
w_l = w_p - 9. * 10 ** 2  # cavity frequency

gamma = 5. * (10 ** 0)  # dissipation from NV [shifted]
# gamma = 0
kappa = w_c / (2.4 * 10 ** 3)  # dissipation from cavity [shifted]
# kappa = 0

omega_l = 1.0 * (10 ** 3)
omega_c = 1.0 * (10 ** 3)

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


def plot_rho_t(t, sol):
    t_printed = []
    for i in t:
        t_printed.append(i * 100)
    fig = plt.figure()
    plt.plot(t_printed, abs(sol[:, 0, 0] ** 2), "orange", label="ee", linestyle='--')
    plt.plot(t_printed, abs(sol[:, 2, 2] ** 2), "blue", label="e+", linewidth=2, linestyle='--')
    plt.plot(t_printed, abs(sol[:, 1, 1] ** 2), "red", label="e-", linewidth=2, linestyle=':')
    plt.plot(t_printed, abs(sol[:, 3, 3] ** 2), "green", label="ge", linewidth=2, linestyle='-')
    plt.plot(t_printed, abs(sol[:, 5, 5] ** 2), "black", label="g+", linestyle=':')
    plt.plot(t_printed, abs(sol[:, 4, 4] ** 2), "gray", label="g-", linestyle='-.')
    fig.suptitle(u'Эволюция заселенностей без диссипаций', fontsize=18)
    plt.xlabel(u't, нс', fontsize=18)
    plt.ylabel(u'Заселенности', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def commutator(a, b):
    return a.dot(b) - b.dot(a)


# Schrodinger equation's
def right_part(y, t):
    hamiltonian = h * (omega_c * (exp(1j * t * (w_m - w_c)) * b.T.dot(a) + exp(-1j * t * (w_m - w_c)) * b.dot(a.T)) +
                       omega_c * (exp(1j * t * (w_p - w_c)) * d.T.dot(a) + exp(-1j * t * (w_p - w_c)) * a.T.dot(d)) +
                       omega_l * (exp(1j * t * (w_m - w_l)) * b.T + exp(-1j * t * (w_m - w_l)) * b) +
                       omega_l * (exp(1j * t * (w_p - w_l)) * d.T + exp(-1j * t * (w_p - w_l)) * d))
    lindblad = - 1j / h * commutator(hamiltonian, y)
    lindblad += gamma / (h * 2.) * (commutator(b.dot(y), b.T) + commutator(b, y.dot(b.T)))
    lindblad += kappa / (h * 2.) * (commutator(a.dot(y), a.T) + commutator(a, y.dot(a.T)))
    return lindblad


def integrate():
    rho_init = array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=complex128)
    t = linspace(0, 5 * 10 ** -3, 2000)
    sol = odeintw(right_part, rho_init, t)
    return t, sol


def main():
    t, sol = integrate()
    plot_rho_t(t, sol)


main()

