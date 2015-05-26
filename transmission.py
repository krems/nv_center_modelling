# -*- coding: utf-8 -*-
__author__ = 'valerii ovchinnikov'

from numpy import *
from matplotlib.pyplot import *

# constants
E = 7.7 * (10 ** -3)  # |-1> |1> distance
D = 2877.0 * (10 ** -3)
h = 1.  # Plank
mu = 2.0028 * 13.99624 * (10 ** 0)  # electron ~ nv
B_z = 0.0  # magnetic field to split |1> |-1> spin states

w_c = 4.706 * (10 ** 5)  # - 2 * 10 ** 4  # cavity frequency
w_nv = w_c

gama = 5. * (10 ** 0)  # dissipation from NV ==> -3
kappa = 5. * (10 ** 2)  # dissipation from cavity ==> 7

omega = kappa / 2.


def reflectance(n, w):
    return 1 - kappa / ((1j * (w_c - w) + kappa / 2.) + n * omega ** 2 / (1j * (w_nv - w) + gama / 2.))


w = linspace(w_c - 6000., w_c + 6000., 10000)
r = []
for i in w:
    r.append(abs(reflectance(1, i)))
plot(w, r)
grid(True)
show()

f = []
for i in w:
    f.append(-1j * log(1. / reflectance(1, i)) / pi)
plot(w, f)
grid(True)
show()