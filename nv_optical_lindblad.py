# -*- coding: utf-8 -*-
__author__ = 'valerii ovchinnikov'

from numpy import *
from odeintw import odeintw
import matplotlib.pyplot as plt

# constants
E = 7.7 * (10 ** -3)  # |-1> |1> distance
D = 2877.0 * (10 ** -3)
h = 1.  # Plank
mu = 2.0028 * 13.99624 * (10 ** 0)  # electron ~ nv
B_z = 0.0  # magnetic field to split |1> |-1> spin states

w_p = 4.706 * (10 ** 5) - mu * B_z  # e<->u transition frequency 10**14
w_m = w_p + E + mu * B_z  # e<->g transition frequency
w_l = w_p  # + 2 * 10 ** 4  # cavity frequency
w_c = w_m  # - 2 * 10 ** 4  # cavity frequency
delta = 0
gama = 5. * (10 ** 2)  # dissipation from NV ==> -3
# gama = 0
kappa = w_c / (2.4 * 10 ** 5)  # dissipation from cavity ==> 7
# kappa = 0

omega_l_m = 1.0 * (10 ** 4)
omega_c_m = 1.0 * (10 ** 4)
omega_l_p = 1.0 * (10 ** 4)
omega_c_p = 1.0 * (10 ** 4)

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


def plot_rho(t, sol):
    plt.pcolor(sol[t.__len__() - 1])
    plt.show()


def plot_rho_t(t, sol):
    # plt.plot(t, abs(sol[:, 0, 0] ** 2), "red", label="ee", linestyle='--')
    plt.plot(t, abs(sol[:, 2, 2] ** 2), "blue", label="e+", linewidth=2.5, linestyle=':')
    plt.plot(t, abs(sol[:, 1, 1] ** 2), "green", label="e-", linewidth=2.5, linestyle='-.')
    # plt.plot(t, abs(sol[:, 3, 3] ** 2), "orange", label="ge", linewidth=2, linestyle='--')
    # plt.plot(t, abs(sol[:, 5, 5] ** 2), "black", label="g+", linestyle=':')
    # plt.plot(t, abs(sol[:, 4, 4] ** 2), "gray", label="g-", linestyle='-.')
    plt.xlabel(u't, нс')
    plt.ylabel(u'Заселенности')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def commutator(a, b):
    return a.dot(b) - b.dot(a)


# Lindblad equation's
def right_part(rho, t):
    ham_c = -h * w_c * a.T.dot(a)
    ham_nv = -h * (delta * b.T.dot(b) + w_p * d.dot(d.T) + w_m * b.dot(b.T))
    ham_nv_c = -h * omega_c_m * (b.T.dot(a) + b.dot(a.T)) - h * omega_c_p * (d.T.dot(a) + d.dot(a.T))
    ham_nv_laser = -h * omega_l_m * (b.T + b) - h * omega_l_p * (d.T + d)
    hamiltonian = ham_c + ham_nv + ham_nv_c + ham_nv_laser

    lindblad = - 1j / h * commutator(hamiltonian, rho)
    lindblad += gama / h / 2. * (commutator(b.dot(rho), b.T) + commutator(b, rho.dot(b.T)))
    lindblad += kappa / h / 2. * (commutator(a.dot(rho), a.T) + commutator(a, rho.dot(a.T)))
    return lindblad


def integrate():
    rho_init = array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=complex128)
    t = linspace(0, 5 * 10 ** -2, 500)
    sol = odeintw(right_part, rho_init, t)
    return t, sol


def main():
    t, sol = integrate()
    plot_rho(t, sol)
    plot_rho_t(t, sol)


main()