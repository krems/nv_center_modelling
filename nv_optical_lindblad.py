__author__ = 'valerii ovchinnikov'

import numpy as np
from odeintw import odeintw
import matplotlib.pyplot as plt

# constants
D = 2877.0 * (10 ** 6)  # g_u distance
E = 7.7 * (10 ** 6)  # |-1> |1> distance
gama = 7.5 * (10 ** 5)  # dissipation
# gama = 0
h = 1  # Plank
mu = 2.0028 * 5.788 * (10 ** -9)  # electron ~ nv

w_eu = 4.8367 * (10 ** 6)  # e->u transition frequency
w_eg = 4.8367 * (10 ** 6) + h * E  # e->g transition frequency
w_p = 4.8 * (10 ** 5)  # cavity frequency
delta = w_eu - w_p  # detuning of cavity relatively to nv resonant transition

B_z = 850.0  # magnetic field to split |1> |-1> spin states
omega_s = 45.0 * (10 ** 6)
omega_p = 100.0 * (10 ** 6)


def plot_rho(t, sol):
    plt.pcolor(sol[t.__len__() - 1])
    plt.show()


def plot_rho_t(t, sol):
    plt.plot(t, abs(sol[:, 0, 0] ** 2), label='0')
    plt.plot(t, abs(sol[:, 1, 1] ** 2), label='-1')
    plt.plot(t, abs(sol[:, 2, 2] ** 2), label='1')
    plt.xlabel('t')
    plt.ylabel('probabilities')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def commutator(a, b):
    return np.dot(a, b) - np.dot(b, a)


def cross(a):
    return np.transpose(- 1 * a)


# Lindblad equation's
def right_part(rho, t):
    hamiltonian = (1. / 2) * np.array(
        [[delta, omega_s, omega_p / 2.0 * np.sin(t * w_p)],
         [omega_s, 0.0, 0.0],
         [omega_p / 2.0 * np.sin(t * w_p), 0.0, 0.0]],
        dtype=np.complex128)
    return commutator(hamiltonian, rho) / 1j + \
           gama * np.dot(
               np.dot(np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]),
                      rho),
               np.array([[0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])) - 0.5 * gama * (np.dot(
               np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]]),
               rho) + np.dot(
               rho,
               np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])))


def integrate():
    rho_init = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]], dtype=np.complex128)
    t = np.linspace(0, 10 ** -5, 2001)
    sol = odeintw(right_part, rho_init, t)
    return t, sol


def main():
    t, sol = integrate()
    plot_rho(t, sol)
    plot_rho_t(t, sol)


main()