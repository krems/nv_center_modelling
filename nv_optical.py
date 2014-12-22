__author__ = 'valerii ovchinnikov'

from numpy import *
from scipy.integrate import ode
import matplotlib.pyplot as plt

# constants
D = 2877.0 * (10 ** 6)  # g_u distance
E = 7.7 * (10 ** 6)  # |-1> |1> distance
# gama = 2.25 * (10 ** 6)  # dissipation
gama = 0
h = 1  # Plank
mu = 2.0028 * 5.788 * (10 ** -9)  # electron ~ nv

w_eu = 4.8367 * (10 ** 6)  # e->u transition frequency
w_eg = 4.8367 * (10 ** 6) + h * D  # e->g transition frequency
w_p = 4.8 * (10 ** 6)  # cavity frequency
delta = w_eu - w_p  # detuning of cavity relatively to nv resonant transition

B_z = 850.0  # magnetic field to split |1> |-1> spin states
# omega_p = 100.0 * (10 ** 6)  # optical field amplitude
omega_p = 0
omega_micr = (h * D + mu * B_z) / h  # microwave field frequency

def B_0(t, A):
    return A * cos(t * 10 ** 9)


def plot_populations(dt, e, g, g2, t0, t1, u):
    x = linspace(t0, t1, (t1 - t0) / dt + 10)
    plt.plot(x, e)
    plt.plot(x, g)
    plt.plot(x, u)
    plt.plot(x, g2)
    plt.show()


def integrate(dt, r, t0, t1):
    e = linspace(t0, t1, (t1 - t0) / dt + 10)
    g = linspace(t0, t1, (t1 - t0) / dt + 10)
    u = linspace(t0, t1, (t1 - t0) / dt + 10)
    g2 = linspace(t0, t1, (t1 - t0) / dt + 10)
    while r.successful() and r.t < t1:
        r.integrate(r.t + dt)
        e[r.t / dt] = abs(r.y[0]) ** 2
        g[r.t / dt] = abs(r.y[1]) ** 2
        u[r.t / dt] = abs(r.y[2]) ** 2
        g2[r.t / dt] = abs(r.y[3]) ** 2
    return e, g, u, g2


# Schrodinger equation's
def right_part(t, y):
    k1 = h * D
    k2 = h * E
    A = 0
    hamiltonian = array(
        [[delta - 1j * gama, omega_p / 2. * sin(t * w_p), 0., 0.],
         [0., k1 + mu * B_z, mu * B_0(t, A) * cos(omega_micr * t), 2. * k2],
         [0., mu * B_0(t, A) * cos(omega_micr * t), 0., mu * B_0(t, A) * cos(omega_micr * t)],
         [omega_p / 2. * sin(t * w_p), 2. * k2, mu * B_0(t, A) * cos(omega_micr * t), k1 - mu * B_z]],
        dtype=complex128)
    hamiltonian *= -1j / h
    return dot(hamiltonian, y)


def create_integrator():
    r = ode(right_part).set_integrator('zvode', method='bdf', with_jacobian=False)
    psi_init = array([0.0, 1.0, 0.0, 0.0], dtype=complex128)
    t0 = 0
    r.set_initial_value(psi_init, t0)
    return r, t0


def main():
    r, t0 = create_integrator()
    t1 = 5 * 10 ** -6
    dt = 10 ** -11
    e, g, u, g2 = integrate(dt, r, t0, t1)
    plot_populations(dt, e, g, g2, t0, t1, u)

main()