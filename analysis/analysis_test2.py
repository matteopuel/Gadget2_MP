import os
import sys

import struct
import numpy as np

import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt

import matplotlib.offsetbox as offsetbox
from matplotlib.legend_handler import HandlerBase

from decimal import *
getcontext().prec = 8 # memorize 8 digits after the comma
from scipy import interpolate
from scipy.optimize import curve_fit 


import readGadget1


plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 18})
rc('text', usetex=True)
#rc('font', size=17)
#rc('legend', fontsize=15)

rc('font', family='serif', size=18)
rc('legend', fontsize=18)
#plt.style.use('classic')


def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def getCoord(vec) :
	vec_x = []
	vec_y = []
	vec_z = []
	for i in range(0, len(vec)) :
		vec_x.append(vec[i][0])
		vec_y.append(vec[i][1])
		vec_z.append(vec[i][2])

	return [vec_x, vec_y, vec_z]


def getRad(x, y, z) :
	return np.sqrt(np.square(x) + np.square(y) + np.square(z))


# GADGET-2 code units
UnitLength_in_cm = 3.085678e21  # = 1 kpc -> a = 1.0 kpc
UnitMass_in_g = 1.989e43  # = 1e10 Msun
UnitVelocity_in_cm_per_s = 1e5  # = 1 km/s -> v0 = 1.0 km/s
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s


# constants from simulation or PDG
h = 0.6732117
Gyr = (365*24*60*60)*1.0e9 # sec
rhocrit = 1.87834e-29 * h**2 # g/cm^3
rhocrit *= (UnitLength_in_cm**3 / UnitMass_in_g) # 1e10 Msun / kpc^3 (code units) 
G = 6.672e-8 # cm^3/g/s^2
G *= (UnitMass_in_g * UnitTime_in_s**2 / UnitLength_in_cm**3) # (code units)


ScatteringCrossSection = 1.0 # cm^2/g (it comes from parameter file!)
ScatteringCrossSection *= (UnitMass_in_g / UnitLength_in_cm**2) # (code units = kpc^2 / (1e10 Msun))



def rhoH(roa, M, a) :
	return M / (2.0 * np.pi) / roa / (1.0 + roa)**3


def veldispH(roa, M, a) :
	fH = 12.0 * roa * (1.0 + roa)**3 * np.log(1.0 + 1.0 / roa) - 1.0 / (1.0 + 1.0 / roa) * (25.0 + 52.0 * roa + 42.0 * roa**2 + 12.0 * roa**3)
	return np.sqrt(G * M / (12.0 * a) * fH)


def GammaH(roa, M, a) :
	vpair = 4.0 / np.sqrt(np.pi) * veldispH(roa, M, a)
	return ScatteringCrossSection * vpair * rhoH(roa, M, a)



# plot y = GammaH(r, M, a) / GammaH(a, M, a)
Xmax = 0.99

##---------# (used for test 2) #---------#
#M = 1.0e5 # * 1e10 Msun (total mass -> galaxy cluster)
#a = 1.0e3 # kpc (scale radius)
#eps = 12.0 # kpc
#basedir = "test2/scattering/benchmark/out/"

#-------# (used for stability) #-------#
M = 1.0e4 # * 1e10 Msun
a = 225.0 # kpc
eps = 4.4 # kpc

M /= Xmax
epsoa = eps / a


#-- these come from the code "analysis_test2_single.py" --#
mid_bins = np.loadtxt(fname='test2/scattering/data/benchmark2.dat', delimiter='\t', usecols = 0) # same mid_bins for all of them!
dmid_bins = np.loadtxt(fname='test2/scattering/data/benchmark2.dat', delimiter='\t', usecols = 2) # same mid_bins for all of them!
Gamma_bins_1eps = np.loadtxt(fname='test2/scattering/data/benchmark2.dat', delimiter='\t', usecols = 1)
dGamma_bins_1eps = np.loadtxt(fname='test2/scattering/data/benchmark2.dat', delimiter='\t', usecols = 3)
Gamma_bins_eps4 = np.loadtxt(fname='test2/scattering/data/benchmark2_eps4.dat', delimiter='\t', usecols = 1)
dGamma_bins_eps4 = np.loadtxt(fname='test2/scattering/data/benchmark2_eps4.dat', delimiter='\t', usecols = 3)
Gamma_bins_2eps = np.loadtxt(fname='test2/scattering/data/benchmark2_2eps.dat', delimiter='\t', usecols = 1)
dGamma_bins_2eps = np.loadtxt(fname='test2/scattering/data/benchmark2_2eps.dat', delimiter='\t', usecols = 3)
Gamma_bins_4eps = np.loadtxt(fname='test2/scattering/data/benchmark2_4eps.dat', delimiter='\t', usecols = 1)
dGamma_bins_4eps = np.loadtxt(fname='test2/scattering/data/benchmark2_4eps.dat', delimiter='\t', usecols = 3)
Gamma_bins_8eps = np.loadtxt(fname='test2/scattering/data/benchmark2_8eps.dat', delimiter='\t', usecols = 1)
dGamma_bins_8eps = np.loadtxt(fname='test2/scattering/data/benchmark2_8eps.dat', delimiter='\t', usecols = 3)


Gamma_th = [GammaH(roa=el, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a) for el in mid_bins]

y_epsoa = GammaH(roa=epsoa, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a)
y_epsoa4 = GammaH(roa=epsoa/4.0, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a)
y_2epsoa = GammaH(roa=2.0*epsoa, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a)
y_4epsoa = GammaH(roa=4.0*epsoa, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a)
y_8epsoa = GammaH(roa=8.0*epsoa, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a)


dGamma_bins_1eps_1p = [(Gamma_bins_1eps[j] + dGamma_bins_1eps[j]) for j in range(0, len(Gamma_bins_1eps))]
dGamma_bins_1eps_1m = [(Gamma_bins_1eps[j] - dGamma_bins_1eps[j]) for j in range(0, len(Gamma_bins_1eps))]
dGamma_bins_eps4_1p = [(Gamma_bins_eps4[j] + dGamma_bins_eps4[j]) for j in range(0, len(Gamma_bins_eps4))]
dGamma_bins_eps4_1m = [(Gamma_bins_eps4[j] - dGamma_bins_eps4[j]) for j in range(0, len(Gamma_bins_eps4))]
dGamma_bins_2eps_1p = [(Gamma_bins_2eps[j] + dGamma_bins_2eps[j]) for j in range(0, len(Gamma_bins_2eps))]
dGamma_bins_2eps_1m = [(Gamma_bins_2eps[j] - dGamma_bins_2eps[j]) for j in range(0, len(Gamma_bins_2eps))]
dGamma_bins_4eps_1p = [(Gamma_bins_4eps[j] + dGamma_bins_4eps[j]) for j in range(0, len(Gamma_bins_4eps))]
dGamma_bins_4eps_1m = [(Gamma_bins_4eps[j] - dGamma_bins_4eps[j]) for j in range(0, len(Gamma_bins_4eps))]
dGamma_bins_8eps_1p = [(Gamma_bins_8eps[j] + dGamma_bins_8eps[j]) for j in range(0, len(Gamma_bins_8eps))]
dGamma_bins_8eps_1m = [(Gamma_bins_8eps[j] - dGamma_bins_8eps[j]) for j in range(0, len(Gamma_bins_8eps))]



##-- Plots --##
fig4 = plt.figure(num='Gamma_vs_roa', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r/a$', fontsize=20)
ax4.set_ylabel(r'$\Gamma_{\rm scatt} (r) / \Gamma_{\rm scatt} (a)$', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(5.7e-3, 2.0)
ax4.set_ylim(1.0e-1, 1.0e3)
ax4.plot(mid_bins, Gamma_th, color ='black', linestyle = '--', lw=2.0, label=r'analytical')

ax4.plot(mid_bins, Gamma_bins_eps4, color='cyan', linestyle='-', lw=2.0, label=r'$h_S = \epsilon/4$')
ax4.fill_between(mid_bins, dGamma_bins_eps4_1m, dGamma_bins_eps4_1p, color ='cyan', alpha=0.3)
ax4.plot(mid_bins, Gamma_bins_1eps, color='blue', linestyle='-', lw=2.0, label=r'$h_S = \epsilon$')
ax4.fill_between(mid_bins, dGamma_bins_1eps_1m, dGamma_bins_1eps_1p, color ='blue', alpha=0.3)
ax4.plot(mid_bins, Gamma_bins_2eps, color='green', linestyle='-', lw=2.0, label=r'$h_S = 2\epsilon$')
ax4.fill_between(mid_bins, dGamma_bins_2eps_1m, dGamma_bins_2eps_1p, color ='green', alpha=0.3)
ax4.plot(mid_bins, Gamma_bins_4eps, color='orange', linestyle='-', lw=2.0, label=r'$h_S = 4\epsilon$')
ax4.fill_between(mid_bins, dGamma_bins_4eps_1m, dGamma_bins_4eps_1p, color ='orange', alpha=0.3)
ax4.plot(mid_bins, Gamma_bins_8eps, color='red', linestyle='-', lw=2.0, label=r'$h_S = 8\epsilon$')
ax4.fill_between(mid_bins, dGamma_bins_8eps_1m, dGamma_bins_8eps_1p, color ='red', alpha=0.3)

ax4.axvline(epsoa, color='gray', linestyle = '-.', lw=2.0)
ax4.plot(epsoa, y_epsoa, color='blue', marker='x', ms=12.0, mew=3.0)
ax4.plot(epsoa/4.0, y_epsoa4, color='cyan', marker='x', ms=12.0, mew=3.0)
ax4.plot(2.0*epsoa, y_2epsoa, color='green', marker='x', ms=12.0, mew=3.0)
ax4.plot(4.0*epsoa, y_4epsoa, color='orange', marker='x', ms=12.0, mew=3.0)
ax4.plot(8.0*epsoa, y_8epsoa, color='red', marker='x', ms=12.0, mew=3.0)
ax4.text(x=epsoa + 2.0e-3, y=3.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()
#fig4.savefig('test2/figs/Gamma_vs_roa2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)




## SECOND PART ##

#-- these come from the code "density_cmp_test2.py" --#
mid_bins_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 0) # same mid_bins for all of them!
dmid_bins_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 1) # same mid_bins for all of them!
rho_bins_i_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 2)
drho_bins_i_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 3)
rho_bins_02_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 4)
drho_bins_02_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 5)
rho_bins_05_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 6)
drho_bins_05_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 7)
rho_bins_07_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 8)
drho_bins_07_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 9)
rho_bins_09_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 10)
drho_bins_09_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 11)
rho_bins_13_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 12)
drho_bins_13_1 = np.loadtxt(fname='test2/stability/data/benchmark2_stability.dat', delimiter='\t', usecols = 13)


r_min = 1.0 # kpc
r_max = 100.0 * a

def rhoHr(r, M, a) :
	return M / (2.0 * np.pi) * (a / r) / (r + a)**3


pt_bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=1000)
rhoH_th = [rhoHr(r=pt_bins[j], M=M*1.0e10, a=a) for j in range(0, len(pt_bins))]


# NORMALIZED quantities #
rhoH_cte = M / (2.0 * np.pi * a**3)
rhoH_cte *= 1.0e10

roa_min = r_min / a
roa_max = r_max / a

ptoa_bins = np.logspace(start=np.log10(roa_min), stop=np.log10(roa_max), num=1000)
rhoHrH_th = [el / rhoH_cte for el in rhoH_th]

mid_bins_1N = [el / a for el in mid_bins_1]
dmid_bins_1N = [el / a for el in dmid_bins_1]
rho_bins_i_1N = [el / rhoH_cte for el in rho_bins_i_1]
drho_bins_i_1N = [el / rhoH_cte for el in drho_bins_i_1]
rho_bins_02_1N = [el / rhoH_cte for el in rho_bins_02_1]
drho_bins_02_1N = [el / rhoH_cte for el in drho_bins_02_1]
rho_bins_05_1N = [el / rhoH_cte for el in rho_bins_05_1]
drho_bins_05_1N = [el / rhoH_cte for el in drho_bins_05_1]
rho_bins_07_1N = [el / rhoH_cte for el in rho_bins_07_1]
drho_bins_07_1N = [el / rhoH_cte for el in drho_bins_07_1]
rho_bins_09_1N = [el / rhoH_cte for el in rho_bins_09_1]
drho_bins_09_1N = [el / rhoH_cte for el in drho_bins_09_1]
rho_bins_13_1N = [el / rhoH_cte for el in rho_bins_13_1]
drho_bins_13_1N = [el / rhoH_cte for el in drho_bins_13_1]



fig5 = plt.figure(num='rho_vs_r_cmp', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlim(1.0e0, 2.9e2)
ax5.set_ylim(0.8e5, 4.0e8)
ax5.errorbar(mid_bins_1, rho_bins_i_1, xerr=0, yerr=drho_bins_i_1, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax5.errorbar(mid_bins_1, rho_bins_02_1, xerr=0, yerr=drho_bins_02_1, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax5.errorbar(mid_bins_1, rho_bins_05_1, xerr=0, yerr=drho_bins_05_1, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax5.errorbar(mid_bins_1, rho_bins_07_1, xerr=0, yerr=drho_bins_07_1, c='yellow', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax5.errorbar(mid_bins_1, rho_bins_09_1, xerr=0, yerr=drho_bins_09_1, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax5.errorbar(mid_bins_1, rho_bins_13_1, xerr=0, yerr=drho_bins_13_1, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax5.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax5.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax5.text(x=eps + 0.5, y=2.0e6, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
#ax5.axvline(2.0*eps, color='pink', linestyle = '-.', lw=2.0)
#ax5.text(x=2.0*eps + 0.5, y=2.0e6, s=r'$2 \epsilon$', rotation=0, color='pink', fontsize=18)
ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.legend(loc='upper right', prop={'size': 18})
ob5 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
ax5.legend(loc='upper right', prop={'size': 18})
#ob5_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
#ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()
#fig5.savefig('test2/figs/rho_vs_r_cmp2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


fig51 = plt.figure(num='rhorH_vs_roa_cmp', figsize=(10, 7), dpi=100)
ax51 = fig51.add_subplot(111)
ax51.set_xlabel(r'$r / a$', fontsize=20)
ax51.set_ylabel(r'$\rho (r) / \rho_H$', fontsize=20)
ax51.set_xscale('log')
ax51.set_yscale('log')
ax51.set_xlim(1.0e0/a, 3.0e2/a)
ax51.set_ylim(0.8e5/rhoH_cte, 4.0e8/rhoH_cte)
ax51.errorbar(mid_bins_1N, rho_bins_i_1N, xerr=0, yerr=drho_bins_i_1N, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax51.errorbar(mid_bins_1N, rho_bins_02_1N, xerr=0, yerr=drho_bins_02_1N, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax51.errorbar(mid_bins_1N, rho_bins_05_1N, xerr=0, yerr=drho_bins_05_1N, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax51.errorbar(mid_bins_1N, rho_bins_07_1N, xerr=0, yerr=drho_bins_07_1N, c='yellow', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax51.errorbar(mid_bins_1N, rho_bins_09_1N, xerr=0, yerr=drho_bins_09_1N, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax51.errorbar(mid_bins_1N, rho_bins_13_1N, xerr=0, yerr=drho_bins_13_1N, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax51.plot(ptoa_bins, rhoHrH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax51.axvline(epsoa, color='gray', linestyle = '-.', lw=2.0)
ax51.text(x=epsoa + 0.5/a, y=2.0e6/rhoH_cte, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
#ax51.axvline(2.0*eps, color='pink', linestyle = '-.', lw=2.0)
#ax51.text(x=2.0*eps + 0.5, y=2.0e6, s=r'$2 \epsilon$', rotation=0, color='pink', fontsize=18)
ax51.grid(False)
ax51.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax51.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax51.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax51.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax51.legend(loc='upper right', prop={'size': 18})
ob51 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob51.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax51.add_artist(ob51)
ax51.legend(loc='upper right', prop={'size': 18})
#ob51_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob51_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
#ax51.add_artist(ob51_1)
fig51.tight_layout()
fig51.show()
#fig51.savefig('test2/figs/rhorH_vs_roa_cmp2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)




#-- these come from the code "density_cmp_test2.py" --#
mid_bins_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 0) # same mid_bins for all of them!
dmid_bins_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 1) # same mid_bins for all of them!
rho_bins_i_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 2)
drho_bins_i_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 3)
rho_bins_02_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 4)
drho_bins_02_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 5)
rho_bins_05_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 6)
drho_bins_05_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 7)
rho_bins_07_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 8)
drho_bins_07_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 9)
rho_bins_09_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 10)
drho_bins_09_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 11)
rho_bins_13_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 12)
drho_bins_13_2 = np.loadtxt(fname='test2/scattering/data/benchmark2_noption_scattering.dat', delimiter='\t', usecols = 13)


# for fit cored Hernquist profile
def rhoCore(r, rcore, beta, M=M*1.0e10, a=a) :
	return M / (2.0 * np.pi) * a / (r**beta + rcore**beta)**(1.0/beta) / (r + a)**3 


coeffrhoCore_02 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_2, ydata=rho_bins_02_2, p0=[0.1*a, 4.0], sigma=drho_bins_02_2)
print "\nfitted [rcore, beta]_02 = ", coeffrhoCore_02[0]
fit_rho_02 = [rhoCore(el, coeffrhoCore_02[0][0], coeffrhoCore_02[0][1]) for el in pt_bins]

coeffrhoCore_05 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_2, ydata=rho_bins_05_2, p0=[0.1*a, 4.0], sigma=drho_bins_05_2)
print "\nfitted [rcore, beta]_05 = ", coeffrhoCore_05[0]
fit_rho_05 = [rhoCore(el, coeffrhoCore_05[0][0], coeffrhoCore_05[0][1]) for el in pt_bins]

coeffrhoCore_07 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_2, ydata=rho_bins_07_2, p0=[0.1*a, 4.0], sigma=drho_bins_07_2)
print "\nfitted [rcore, beta]_07 = ", coeffrhoCore_07[0]
fit_rho_07 = [rhoCore(el, coeffrhoCore_07[0][0], coeffrhoCore_07[0][1]) for el in pt_bins]

coeffrhoCore_09 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_2, ydata=rho_bins_09_2, p0=[0.1*a, 4.0], sigma=drho_bins_09_2)
print "\nfitted [rcore, beta]_09 = ", coeffrhoCore_09[0]
fit_rho_09 = [rhoCore(el, coeffrhoCore_09[0][0], coeffrhoCore_09[0][1]) for el in pt_bins]

coeffrhoCore_13 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_2, ydata=rho_bins_13_2, p0=[0.1*a, 4.0], sigma=drho_bins_13_2)
print "\nfitted [rcore, beta]_13 = ", coeffrhoCore_13[0]
fit_rho_13 = [rhoCore(el, coeffrhoCore_13[0][0], coeffrhoCore_13[0][1]) for el in pt_bins]



# NORMALIZE quantities #
mid_bins_2N = [el / a for el in mid_bins_2]
dmid_bins_2N = [el / a for el in dmid_bins_2]
rho_bins_i_2N = [el / rhoH_cte for el in rho_bins_i_2]
drho_bins_i_2N = [el / rhoH_cte for el in drho_bins_i_2]
rho_bins_02_2N = [el / rhoH_cte for el in rho_bins_02_2]
drho_bins_02_2N = [el / rhoH_cte for el in drho_bins_02_2]
rho_bins_05_2N = [el / rhoH_cte for el in rho_bins_05_2]
drho_bins_05_2N = [el / rhoH_cte for el in drho_bins_05_2]
rho_bins_07_2N = [el / rhoH_cte for el in rho_bins_07_2]
drho_bins_07_2N = [el / rhoH_cte for el in drho_bins_07_2]
rho_bins_09_2N = [el / rhoH_cte for el in rho_bins_09_2]
drho_bins_09_2N = [el / rhoH_cte for el in drho_bins_09_2]
rho_bins_13_2N = [el / rhoH_cte for el in rho_bins_13_2]
drho_bins_13_2N = [el / rhoH_cte for el in drho_bins_13_2]

fit_rho_02N = [el / rhoH_cte for el in fit_rho_02]
fit_rho_05N = [el / rhoH_cte for el in fit_rho_05]
fit_rho_07N = [el / rhoH_cte for el in fit_rho_07]
fit_rho_09N = [el / rhoH_cte for el in fit_rho_09]
fit_rho_13N = [el / rhoH_cte for el in fit_rho_13]



fig6 = plt.figure(num='rho_vs_r_cmp_scatt', figsize=(10, 7), dpi=100)
ax6 = fig6.add_subplot(111)
ax6.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax6.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlim(1.0e0, 2.9e2)
ax6.set_ylim(0.8e5, 4.0e8)
ax6.errorbar(mid_bins_2, rho_bins_i_2, xerr=0, yerr=drho_bins_i_2, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax6.errorbar(mid_bins_2, rho_bins_02_2, xerr=0, yerr=drho_bins_02_2, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax6.errorbar(mid_bins_2, rho_bins_05_2, xerr=0, yerr=drho_bins_05_2, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax6.errorbar(mid_bins_2, rho_bins_07_2, xerr=0, yerr=drho_bins_07_2, c='yellow', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax6.errorbar(mid_bins_2, rho_bins_09_2, xerr=0, yerr=drho_bins_09_2, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax6.errorbar(mid_bins_2, rho_bins_13_2, xerr=0, yerr=drho_bins_13_2, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax6.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax6.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax6.text(x=eps + 0.5, y=2.0e6, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax6.plot(pt_bins, fit_rho_02, color ='cyan', linestyle = '-', lw=1.5)
ax6.plot(pt_bins, fit_rho_05, color ='green', linestyle = '-', lw=1.5)
ax6.plot(pt_bins, fit_rho_07, color ='yellow', linestyle = '-', lw=1.5)
ax6.plot(pt_bins, fit_rho_09, color ='orange', linestyle = '-', lw=1.5)
ax6.plot(pt_bins, fit_rho_13, color ='red', linestyle = '-', lw=1.5)
ax6.grid(False)
ax6.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax6.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax6.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax6.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax6.legend(loc='upper right', prop={'size': 18})
ob6 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob6.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax6.add_artist(ob6)
ax6.legend(loc='upper right', prop={'size': 18})
ob6_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob6_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax6.add_artist(ob6_1)
fig6.tight_layout()
fig6.show()
#fig6.savefig('test2/figs/rho_vs_r_cmp_scatt2_1.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


fig61 = plt.figure(num='rhorH_vs_roa_cmp_scatt', figsize=(10, 7), dpi=100)
ax61 = fig61.add_subplot(111)
ax61.set_xlabel(r'$r / a$', fontsize=20)
ax61.set_ylabel(r'$\rho (r) / \rho_H$', fontsize=20)
ax61.set_xscale('log')
ax61.set_yscale('log')
ax61.set_xlim(1.0e0/a, 3.0e2/a)
ax61.set_ylim(0.8e5/rhoH_cte, 4.0e8/rhoH_cte)
ax61.errorbar(mid_bins_2N, rho_bins_i_2N, xerr=0, yerr=drho_bins_i_2N, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax61.errorbar(mid_bins_2N, rho_bins_02_2N, xerr=0, yerr=drho_bins_02_2N, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax61.errorbar(mid_bins_2N, rho_bins_05_2N, xerr=0, yerr=drho_bins_05_2N, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax61.errorbar(mid_bins_2N, rho_bins_07_2N, xerr=0, yerr=drho_bins_07_2N, c='yellow', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax61.errorbar(mid_bins_2N, rho_bins_09_2N, xerr=0, yerr=drho_bins_09_2N, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax61.errorbar(mid_bins_2N, rho_bins_13_2N, xerr=0, yerr=drho_bins_13_2N, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax61.plot(ptoa_bins, rhoHrH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax61.axvline(epsoa, color='gray', linestyle = '-.', lw=2.0)
ax61.text(x=epsoa + 0.5/a, y=2.0e6/rhoH_cte, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax61.plot(ptoa_bins, fit_rho_02N, color ='cyan', linestyle = '-', lw=1.5)
ax61.plot(ptoa_bins, fit_rho_05N, color ='green', linestyle = '-', lw=1.5)
ax61.plot(ptoa_bins, fit_rho_07N, color ='yellow', linestyle = '-', lw=1.5)
ax61.plot(ptoa_bins, fit_rho_09N, color ='orange', linestyle = '-', lw=1.5)
ax61.plot(ptoa_bins, fit_rho_13N, color ='red', linestyle = '-', lw=1.5)
ax61.grid(False)
ax61.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax61.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax61.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax61.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax61.legend(loc='upper right', prop={'size': 18})
ob61 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob61.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax61.add_artist(ob61)
ax61.legend(loc='upper right', prop={'size': 18})
ob61_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob61_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax61.add_artist(ob61_1)
fig61.tight_layout()
fig61.show()
#fig61.savefig('test2/figs/rhorH_vs_roa_cmp_scatt2_1.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)




raw_input()
