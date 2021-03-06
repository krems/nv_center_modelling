# -*- coding: utf-8 -*-
__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
import matplotlib.pyplot as plt

# constants
h = 1.  # Plank
E = 7.7 * (10 ** -3)  # ground m_s=-1 <-> m_s=1 splitting
D = 2877.0 * (10 ** -3)  # ground m_s=0 <-> m_s=+-1 splitting
D_e = 1420.0 * (10 ** -3)  # excited state m_s=0 <-> m_s=+-1 splitting
delta = 0
mu = 2.0028 * 13.99624 * (10 ** 0)  # electron ~ nv
B_z = 0.0  # magnetic field to split |1> |-1> spin states

w_p = 4.706 * (10 ** 5) - mu * B_z  # e<->u transition frequency 10**14
w_m = 4.706 * (10 ** 5) + E + mu * B_z  # e<->g transition frequency
w_z = 4.706 * (10 ** 5) + D

omega_m = 100.0 * (10 ** 4)  # Rabi frequency
omega_p = 100.0 * (10 ** 4)  # Rabi frequency
omega_z = 100.0 * (10 ** 4)  # Rabi frequency
omega_parasite = omega_m / 10.0  # Rabi frequency
omega_relaxation = omega_m / 10.0  # Rabi frequency

# basis vectors
e = mat(array([1., 0., 0., 0., 0.], dtype=complex128))
ez = mat(array([0., 1., 0., 0., 0.], dtype=complex128))
p = mat(array([0., 0., 1., 0., 0.], dtype=complex128))
m = mat(array([0., 0., 0., 1., 0.], dtype=complex128))
z = mat(array([0., 0., 0., 0., 1.], dtype=complex128))


def plot_populations(e, ez, p, m, z, t0, t1, dt):
    fig = plt.figure()
    x = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(x, e, "green", label="e", linewidth=2)
    plt.plot(x, ez, "orange", label="ez", linestyle='--')
    plt.plot(x, p, "blue", label="+", linewidth=2, linestyle='--')
    plt.plot(x, m, "red", label="-", linewidth=2, linestyle=':')
    plt.plot(x, z, "black", label="0", linestyle='-')
    fig.suptitle(u'Эволюция заселенностей с учетом паразитных уровней', fontsize=18)
    plt.xticks([])
    plt.xlabel(u't', fontsize=18)
    plt.ylabel(u'Заселенности', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def integrate(integrator, t0, t1, dt):
    e = linspace(t0, t1, (t1 - t0) / dt + 10)
    ez = linspace(t0, t1, (t1 - t0) / dt + 10)
    p = linspace(t0, t1, (t1 - t0) / dt + 10)
    m = linspace(t0, t1, (t1 - t0) / dt + 10)
    z = linspace(t0, t1, (t1 - t0) / dt + 10)
    while integrator.successful() and integrator.t < t1:
        integrator.integrate(integrator.t + dt)
        e[integrator.t / dt] = abs(integrator.y[0]) ** 2
        ez[integrator.t / dt] = abs(integrator.y[1]) ** 2
        p[integrator.t / dt] = abs(integrator.y[2]) ** 2
        m[integrator.t / dt] = abs(integrator.y[3]) ** 2
        z[integrator.t / dt] = abs(integrator.y[4]) ** 2
    return e, ez, p, m, z


# Schrodinger equation's
def right_part(t, y):
    hamiltonian = -delta * e.T.dot(e) - D_e * ez.T.dot(ez) - w_p * p.T.dot(p) - w_m * m.T.dot(m) - w_z * z.T.dot(z)
    hamiltonian += omega_m * (e.T.dot(m) + m.T.dot(e)) + omega_p * (e.T.dot(p) + p.T.dot(e)) + \
                   omega_z * (ez.T.dot(z) + z.T.dot(ez)) + omega_parasite * (ez.T.dot(p) + ez.T.dot(m) + p.T.dot(ez) + m.T.dot(ez)) + \
                   omega_relaxation * (z.T.dot(e))
    hamiltonian *= -1j
    return dot(hamiltonian, y)


def create_integrator():
    integrator = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([0.0, 0.0, 1.0, 0.0, 0.0],
                     dtype=complex128)
    t0 = 0
    integrator.set_initial_value(psi_init, t0)
    return integrator, t0


def main():
    integrator, t0 = create_integrator()
    t1 = 5 * 10 ** -5
    dt = 10 ** -8
    e, ez, p, m, z = integrate(integrator, t0, t1, dt)
    plot_populations(e, ez, p, m, z, t0, t1, dt)


main()
