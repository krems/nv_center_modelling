# -*- coding: utf-8 -*-
__author__ = 'valerii ovchinnikov'

from numpy import *
from matplotlib.pyplot import *

w_c = 4.706 * (10 ** 5)  # - 2 * 10 ** 4  # cavity frequency
w_nv = w_c

gamma = 5. * (10 ** -2)  # dissipation from NV
kappa = w_c / (2.4 * 10 ** 5)  # dissipation from cavity

omega = kappa / 2.


def reflectance(w):
    return 1. - kappa / (1j * (w_c - w) + kappa / 2. + omega ** 2 / (1j * (w_nv - w) + gamma / 2.))

def transmission(omega, w):
    return 1. - kappa / (1j * (w_c - w) + kappa / 2. + omega**2 / (1j * (w_nv - w) - gamma / 2.))

w = linspace(w_c - 5, w_c + 5, 10000)
x = []
r = []
e = []
for i in w:
    r.append(abs(1. / transmission(omega, i)))
    e.append(abs(transmission(0, i)))
    x.append((i - w_c) / kappa)

fig = figure()
fig.suptitle(u'Коэффициент отражения', fontsize=18)
xlabel(u'(w - w)', fontsize=18)
ylabel(u'Коэффициент отражения (1 / T)', fontsize=18)
plot(x, r, "blue", label=u"С NV-центром", linewidth=2, linestyle='--')
plot(x, e, "red", label=u"Пустой", linewidth=2, linestyle='-')
grid(True)
legend(loc='best')
show()

f = []
fe = []
for i in w:
    f.append(1j * log(transmission(omega, i)) / pi)
    fe.append(1j * log(transmission(0, i)) / pi)
fig = figure()
fig.suptitle(u'Сдвиг фазы', fontsize=18)
xlabel(u'(w - w)', fontsize=18)
ylabel(u'Сдвиг фазы / п ( )', fontsize=18)
plot(x, f, "blue", label=u"С NV-центром", linewidth=2, linestyle='--')
plot(x, fe, "red", label=u"Пустой", linewidth=2, linestyle='-')
grid(True)
legend(loc='best')
show()

