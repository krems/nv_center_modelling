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
gama = 5. * (10 ** 2)  # dissipation from NV [shifted]
# gama = 0
kappa = w_c / (2.4 * 10 ** 2)  # dissipation from cavity [shifted]
# kappa = 0

omega_l_m = 1.0 * (10 ** 4)  # [shifted]
omega_c_m = 1.0 * (10 ** 4)  # [shifted]
omega_l_p = 1.0 * (10 ** 4)  # [shifted]
omega_c_p = 1.0 * (10 ** 4)  # [shifted]

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

def f(x):
    c = (w_p + w_m) / 2.
    b = omega_c_m
    a = omega_c_p
    d = 4.*a**2 + 4.*b**2 + 9*c**2
    return b**2/(a**2+b**2)*exp(-c*x) + ((3.* a* c)/sqrt(d)+a)/(4.* (a**2+b**2))*2.*a*exp((c-sqrt(d))*x/2.) + (a-(3.* a *c)/sqrt(d))/(4.* (a**2+b**2))*2.*a*exp((c+sqrt(d)*t/2.))

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
