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
gama = 5. * (10 ** 0)  # dissipation from NV ==> -3
# gama = 0
kappa = w_c / (2.4 * 10 ** 3)  # dissipation from cavity ==> 7
# kappa = 0

omega_l = 100.0 * (10 ** 3)
omega_c = 100.0 * (10 ** 3)

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
    plt.plot(t, abs(sol[:, 0, 0] ** 2), label='ee')
    plt.plot(t, abs(sol[:, 1, 1] ** 2), label='e-1', linewidth=2)
    plt.plot(t, abs(sol[:, 2, 2] ** 2), label='e1', linewidth=2)
    plt.plot(t, abs(sol[:, 3, 3] ** 2), label='ge', linewidth=2)
    plt.plot(t, abs(sol[:, 4, 4] ** 2), label='g-1')
    plt.plot(t, abs(sol[:, 5, 5] ** 2), label='g1')
    plt.xlabel('t')
    plt.ylabel('probabilities')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def commutator(a, b):
    return a.dot(b) - b.dot(a)


# Lindblad equation's
def right_part(rho, t):
    ham_c = -h * w_c * (a.T.dot(a) + kron(E_two, E_three) * 1. / 2.)
    ham_nv = -h * (delta * b.T.dot(b) + w_p * d.dot(d.T) + w_m * b.dot(b.T))
    ham_nv_c = -h / 2. * omega_c * (exp(-1j * (w_m - w_c) * t) * b.T.dot(a) + exp(1j * (w_m - w_c) * t) * b.dot(a.T)
               + exp(-1j * (w_p - w_c) * t) * d.T.dot(a) + exp(1j * (w_p - w_c) * t) * d.dot(a.T))
    ham_nv_laser = -h / 2. * omega_l * (exp(-1j * (w_m - w_l) * t) * b.T + exp(1j * (w_m - w_l) * t) * b
                   + exp(-1j * (w_p - w_l) * t) * d.T + exp(1j * (w_p - w_l) * t) * d)
    hamiltonian = ham_c + ham_nv + ham_nv_c + ham_nv_laser

    lindblad = - 1j / h * commutator(rho, hamiltonian)
    lindblad += gama * b.dot(rho.dot(b.T)) - gama / 2.0 * commutator(b.T.dot(b), rho)
    lindblad += kappa * a.dot(rho.dot(a.T)) - kappa / 2.0 * commutator(a.T.dot(a), rho)
    return lindblad


def integrate():
    rho_init = array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=complex128)
    t = linspace(0, 9 * 10 ** -3, 2000)
    sol = odeintw(right_part, rho_init, t)
    return t, sol


def main():
    t, sol = integrate()
    plot_rho(t, sol)
    plot_rho_t(t, sol)


main()