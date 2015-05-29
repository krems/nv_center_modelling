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

w_p = 4.706 * (10 ** 5) - mu * B_z  # e<->u transition frequency 10**14
w_m = w_p + E + mu * B_z  # e<->g transition frequency
w_c = w_m  # - 2 * 10 ** 4  # cavity frequency
delta = 0
gamma = 5. * (10 ** 1)  # dissipation from NV
Q = 2.4 * 10 ** 4  # quality of cavity
kappa = w_c / Q  # dissipation from cavity

omega_l_m = 1. * (10 ** 4)  # Rabi frequency
omega_c_m = 1. * (10 ** 4)  # Rabi frequency
omega_l_p = 1. * (10 ** 4)  # Rabi frequency
omega_c_p = 1. * (10 ** 4)  # Rabi frequency

cav_g = mat(array([0.0, 1.0]), dtype=complex128)
cav_e = mat(array([1.0, 0.0]), dtype=complex128)

nv_e = mat(array([1.0, 0.0, 0.0]), dtype=complex128)
nv_p = mat(array([0.0, 1.0, 0.0]), dtype=complex128)
nv_m = mat(array([0.0, 0.0, 1.0]), dtype=complex128)

E_two = matrix([[1.0, 0.0],
                [0.0, 1.0]], dtype=complex128)
E_three = matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=complex128)

b = kron(E_two, nv_m.T.dot(nv_e))
d = kron(E_two, nv_p.T.dot(nv_e))
a = kron(cav_g.T.dot(cav_e), E_three)


def plot_rho(t, sol):
    rho_sq = sol[t]
    for i in range(0, 6):
        for j in range(0, 6):
            rho_sq[i][j] = abs(sol[t][i][j]) ** 2
    print(rho_sq)
    plt.pcolor(rho_sq)
    plt.colorbar()
    plt.show()


def plot_rho_t(t, sol):
    t_printed = []
    for i in t:
        t_printed.append(i * 1000)
    fig = plt.figure()
    # plt.plot(t_printed, abs(sol[:, 0, 0] ** 2), "orange", label="ee", linestyle='--')
    plt.plot(t_printed, abs(sol[:, 2, 2] ** 2), "blue", label="e+", linewidth=2, linestyle='--')
    plt.plot(t_printed, abs(sol[:, 1, 1] ** 2), "red", label="e-", linewidth=2, linestyle=':')
    # plt.plot(t_printed, abs(sol[:, 3, 3] ** 2), "green", label="ge", linewidth=2, linestyle='-')
    # plt.plot(t_printed, abs(sol[:, 5, 5] ** 2), "black", label="g+", linestyle=':')
    # plt.plot(t_printed, abs(sol[:, 4, 4] ** 2), "gray", label="g-", linestyle='-.')
    fig.suptitle(u'Эволюция заселенностей без диссипаций', fontsize=18)
    plt.xlabel(u't, мкс', fontsize=18)
    plt.ylabel(u'Заселенности', fontsize=18)
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
    lindblad += gamma / (h * 2.) * (commutator(b.dot(rho), b.T) + commutator(b, rho.dot(b.T)))
    lindblad += kappa / (h * 2.) * (commutator(a.dot(rho), a.T) + commutator(a, rho.dot(a.T)))
    return lindblad


def integrate():
    rho_init = array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=complex128)
    t = linspace(0, 7 * 10 ** -2, 2000)
    sol = odeintw(right_part, rho_init, t)
    return t, sol


def main():
    t, sol = integrate()
    plot_rho_t(t, sol)


main()
