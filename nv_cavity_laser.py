__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
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


def plot_populations(dt, ee, eg, eu, ge, gg, gu, t0, t1):
    x = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(x, ee, "red", label="ee", linestyle='--')
    plt.plot(x, eg, "blue", label="e+")
    plt.plot(x, eu, "green", label="e-")
    plt.plot(x, ge, "orange", label="ge", linewidth=2, linestyle='--')
    plt.plot(x, gg, "black", label="g+", linewidth=2)
    plt.plot(x, gu, "gray", label="g-", linewidth=2)
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
    ham_c = -h * w_c * (a.T.dot(a) + kron(E_two, E_three) * 1. / 2.)
    ham_nv = -h * (delta * b.T.dot(b) + w_p * d.dot(d.T) + w_m * b.dot(b.T))
    ham_nv_c = -h / 2. * omega_c * (exp(-1j * (w_m - w_c) * t) * b.T.dot(a) + exp(1j * (w_m - w_c) * t) * b.dot(a.T)
               + exp(-1j * (w_p - w_c) * t) * d.T.dot(a) + exp(1j * (w_p - w_c) * t) * d.dot(a.T))
    ham_nv_laser = -h / 2. * omega_l * (exp(-1j * (w_m - w_l) * t) * b.T + exp(1j * (w_m - w_l) * t) * b
                   + exp(-1j * (w_p - w_l) * t) * d.T + exp(1j * (w_p - w_l) * t) * d)
    hamiltonian = ham_c + ham_nv + ham_nv_c + ham_nv_laser - 1j * gama * b.T.dot(b) - 1j * kappa * a.T.dot(a)
    return dot(hamiltonian, y) * (-1j / h)


def create_integrator():
    r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex128)
    t0 = 0.
    r.set_initial_value(psi_init, t0)
    return r, t0


def main():
    r, t0 = create_integrator()
    t1 = 5 * 10 ** -3
    dt = 5 * 10 ** -8
    ee, eg, eu, ge, gg, gu = integrate(dt, r, t0, t1)
    plot_populations(dt, ee, eg, eu, ge, gg, gu, t0, t1)


main()