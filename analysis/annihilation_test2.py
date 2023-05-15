import os.path
import sys
from itertools import chain

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
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # [s * kpc / km]


# constants from simulation or PDG
h = 0.6732117
Gyr = (365*24*60*60)*1.0e9 # sec
rhocrit = 1.87834e-29 * h**2 # g/cm^3
rhocrit *= (UnitLength_in_cm**3 / UnitMass_in_g) # 1e10 Msun / kpc^3 (code units) 
G = 6.672e-8 # cm^3/g/s^2
G *= (UnitMass_in_g * UnitTime_in_s**2 / UnitLength_in_cm**3) # (code units)


sigmav_ann = 100.0 # cm^2/g * km/s
sigmav_ann *= UnitMass_in_g / UnitLength_in_cm**2 * 1.0e5 / UnitVelocity_in_cm_per_s # (code units)
print "sigmav_ann = ", sigmav_ann


def rhoH(r, M, a) :
	return M / (2.0 * np.pi) * (a / r) / (r + a)**3


def rhoHa(roa, M, a) :
	return M / (2.0 * np.pi) / roa / (1.0 + roa)**3

def GammaH(roa, M, a) :
	return sigmav_ann * rhoHa(roa, M, a)



# Hernquist and simulation parameters in code units!
#-------# (used for stability) #-------#
#infile_ini = "test2/hernquist_test2_v2"
#infile_ini = "test2/stability/benchmark2/out/snp_013"
M = 1.0e4 # * 1e10 Msun
a = 225.0 # kpc
eps = 4.4 # kpc


Xmax = 0.99
M /= Xmax # the mass is a bit larger if we want that M(<r_max) = mass we want
epsoa = eps / a



#-- these come from the code "analysis_test2_single.py" --#
mid_bins1 = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION.dat', delimiter='\t', usecols = 0) # same mid_bins for all of them!
dmid_bins1 = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION.dat', delimiter='\t', usecols = 2) # same mid_bins for all of them!
Gamma_bins_1eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION.dat', delimiter='\t', usecols = 1)
dGamma_bins_1eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION.dat', delimiter='\t', usecols = 3)
Gamma_bins_eps4 = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_eps4.dat', delimiter='\t', usecols = 1)
dGamma_bins_eps4 = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_eps4.dat', delimiter='\t', usecols = 3)
Gamma_bins_2eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_2eps.dat', delimiter='\t', usecols = 1)
dGamma_bins_2eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_2eps.dat', delimiter='\t', usecols = 3)
Gamma_bins_4eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_4eps.dat', delimiter='\t', usecols = 1)
dGamma_bins_4eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_4eps.dat', delimiter='\t', usecols = 3)
Gamma_bins_8eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_8eps.dat', delimiter='\t', usecols = 1)
dGamma_bins_8eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_OPTION_8eps.dat', delimiter='\t', usecols = 3)


Gamma_th = [GammaH(roa=el, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a) for el in mid_bins1]

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
fig6 = plt.figure(num='Gamma_vs_roa2_ann', figsize=(10, 7), dpi=100)
ax6 = fig6.add_subplot(111)
ax6.set_xlabel(r'$r/a$', fontsize=20)
ax6.set_ylabel(r'$\Gamma_{\rm ann} (r) / \Gamma_{\rm ann} (a)$', fontsize=20)
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlim(5.7e-3, 2.5)
ax6.set_ylim(1.0e-1, 2.0e3)
ax6.plot(mid_bins1, Gamma_th, color ='black', linestyle = '--', lw=2.0, label=r'analytical')

ax6.plot(mid_bins1, Gamma_bins_eps4, color='cyan', linestyle='-', lw=2.0, label=r'$h_A = \epsilon/4$')
ax6.fill_between(mid_bins1, dGamma_bins_eps4_1m, dGamma_bins_eps4_1p, color ='cyan', alpha=0.3)
ax6.plot(mid_bins1, Gamma_bins_1eps, color='blue', linestyle='-', lw=2.0, label=r'$h_A = \epsilon$')
ax6.fill_between(mid_bins1, dGamma_bins_1eps_1m, dGamma_bins_1eps_1p, color ='blue', alpha=0.3)
ax6.plot(mid_bins1, Gamma_bins_2eps, color='green', linestyle='-', lw=2.0, label=r'$h_A = 2\epsilon$')
ax6.fill_between(mid_bins1, dGamma_bins_2eps_1m, dGamma_bins_2eps_1p, color ='green', alpha=0.3)
ax6.plot(mid_bins1, Gamma_bins_4eps, color='orange', linestyle='-', lw=2.0, label=r'$h_A = 4\epsilon$')
ax6.fill_between(mid_bins1, dGamma_bins_4eps_1m, dGamma_bins_4eps_1p, color ='orange', alpha=0.3)
ax6.plot(mid_bins1, Gamma_bins_8eps, color='red', linestyle='-', lw=2.0, label=r'$h_A = 8\epsilon$')
ax6.fill_between(mid_bins1, dGamma_bins_8eps_1m, dGamma_bins_8eps_1p, color ='red', alpha=0.3)

ax6.axvline(epsoa, color='gray', linestyle = '-.', lw=2.0)
ax6.plot(epsoa, y_epsoa, color='blue', marker='x', ms=12.0, mew=3.0)
ax6.plot(epsoa/4.0, y_epsoa4, color='cyan', marker='x', ms=12.0, mew=3.0)
ax6.plot(2.0*epsoa, y_2epsoa, color='green', marker='x', ms=12.0, mew=3.0)
ax6.plot(4.0*epsoa, y_4epsoa, color='orange', marker='x', ms=12.0, mew=3.0)
ax6.plot(8.0*epsoa, y_8epsoa, color='red', marker='x', ms=12.0, mew=3.0)
ax6.text(x=epsoa + 2.0e-3, y=3.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax6.grid(False)
ax6.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax6.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax6.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax6.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax6.legend(loc='upper right', prop={'size': 18})
ob6 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob6.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax6.add_artist(ob6)
ob6_1 = offsetbox.AnchoredText(s=r'$\langle \sigma_{\rm ann} v \rangle /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob6_1 = offsetbox.AnchoredText(s=r'$\langle \sigma v \rangle_a /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob6_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax6.add_artist(ob6_1)
fig6.tight_layout()
fig6.show()
#fig6.savefig('test2/figs/Gamma_vs_roa2_ann.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)




## SECOND PART ##


#-- these come from the code "annihilation_test2_single.py" --#
mid_bins = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption.dat', delimiter='\t', usecols = 0) # same mid_bins for all of them!
dmid_bins = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption.dat', delimiter='\t', usecols = 2) # same mid_bins for all of them!
rho_bins_1eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption.dat', delimiter='\t', usecols = 1)
drho_bins_1eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption.dat', delimiter='\t', usecols = 3)
rho_bins_eps4 = np.loadtxt(fname='test2/annihilation/data/benchmark2_eps4.dat', delimiter='\t', usecols = 1)
drho_bins_eps4 = np.loadtxt(fname='test2/annihilation/data/benchmark2_eps4.dat', delimiter='\t', usecols = 3)
rho_bins_2eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_2eps.dat', delimiter='\t', usecols = 1)
drho_bins_2eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_2eps.dat', delimiter='\t', usecols = 3)
rho_bins_4eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_4eps.dat', delimiter='\t', usecols = 1)
drho_bins_4eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_4eps.dat', delimiter='\t', usecols = 3)
rho_bins_8eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_8eps.dat', delimiter='\t', usecols = 1)
drho_bins_8eps = np.loadtxt(fname='test2/annihilation/data/benchmark2_8eps.dat', delimiter='\t', usecols = 3)


time = 9.99999999069 # Gyr


r_min = mid_bins[0] - dmid_bins[0]
r_max = mid_bins[-1] - dmid_bins[-1]


pt_bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=1000)
rhoH_th = [rhoH(r=el, M=M*1.0e10, a=a) for el in pt_bins]

rhoH_cte = M / (2.0 * np.pi * a**3)
rhoH_cte *= 1.0e10



# for annihilation-cored Hernquist profile
def rhoH_ann(r, M=M*1.0e10, a=a, time=0.0) :
	x = r / a
	rho_core = 1.0 / (sigmav_ann * time) / 2.0 # (code units) --> 2 units of mass disappears, not just one!
	rho_core *= 1.0e10 # since we want [Msun / kpc^3]
	if r == r_min :
		print "\nrho_core = ", rho_core
	return rhoH_cte / (x * (1.0 + x)**3 + rhoH_cte / rho_core)

fit_rhoH_ann = [rhoH_ann(r=el, M=M*1.0e10, a=a, time=time) for el in pt_bins]
rho_core_true = 1.0 / (sigmav_ann * time) * 1.0e10 # Msun / kpc^3


def rhoH_ann_core(r, rho_core, M=M*1.0e10, a=a) :
	x = r / a
	return rhoH_cte / (x * (1.0 + x)**3 + rhoH_cte / rho_core)


mid_bins_sub_1eps = []
rho_bins_sub_1eps = []
drho_bins_sub_1eps = []

mid_bins_sub_eps4 = []
rho_bins_sub_eps4 = []
drho_bins_sub_eps4 = []

mid_bins_sub_2eps = []
rho_bins_sub_2eps = []
drho_bins_sub_2eps = []

mid_bins_sub_4eps = []
rho_bins_sub_4eps = []
drho_bins_sub_4eps = []

mid_bins_sub_8eps = []
rho_bins_sub_8eps = []
drho_bins_sub_8eps = []

for j in range(0, len(mid_bins)) :
	if rho_bins_1eps[j] >= rho_bins_1eps[-1] : # since rho(r) is monotonically decreasing function
		mid_bins_sub_1eps.append(mid_bins[j])
		rho_bins_sub_1eps.append(rho_bins_1eps[j])
		drho_bins_sub_1eps.append(drho_bins_1eps[j])

	if rho_bins_eps4[j] >= rho_bins_eps4[-1] :
		mid_bins_sub_eps4.append(mid_bins[j])
		rho_bins_sub_eps4.append(rho_bins_eps4[j])
		drho_bins_sub_eps4.append(drho_bins_eps4[j])

	if rho_bins_2eps[j] >= rho_bins_2eps[-1] :
		mid_bins_sub_2eps.append(mid_bins[j])
		rho_bins_sub_2eps.append(rho_bins_2eps[j])
		drho_bins_sub_2eps.append(drho_bins_2eps[j])

	if rho_bins_4eps[j] >= rho_bins_4eps[-1] :
		mid_bins_sub_4eps.append(mid_bins[j])
		rho_bins_sub_4eps.append(rho_bins_4eps[j])
		drho_bins_sub_4eps.append(drho_bins_4eps[j])

	if rho_bins_8eps[j] >= rho_bins_8eps[-1] :
		mid_bins_sub_8eps.append(mid_bins[j])
		rho_bins_sub_8eps.append(rho_bins_8eps[j])
		drho_bins_sub_8eps.append(drho_bins_8eps[j])

coeffrhoH_ann_core_1eps = curve_fit(f=lambda r, rho_core: rhoH_ann_core(r, rho_core), xdata=mid_bins_sub_1eps, ydata=rho_bins_sub_1eps, p0=[rho_core_true], sigma=drho_bins_sub_1eps)
print "fitted rho_core_1eps = ", coeffrhoH_ann_core_1eps[0][0], " +- ", np.sqrt(coeffrhoH_ann_core_1eps[1][0][0])
fit_rhoH_ann_core_1eps = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_1eps[0][0], M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_1eps_1p = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_1eps[0][0]+np.sqrt(coeffrhoH_ann_core_1eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_1eps_1m = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_1eps[0][0]-np.sqrt(coeffrhoH_ann_core_1eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]

coeffrhoH_ann_core_eps4 = curve_fit(f=lambda r, rho_core: rhoH_ann_core(r, rho_core), xdata=mid_bins_sub_eps4, ydata=rho_bins_sub_eps4, p0=[rho_core_true], sigma=drho_bins_sub_eps4)
print "fitted rho_core_eps4 = ", coeffrhoH_ann_core_eps4[0][0], " +- ", np.sqrt(coeffrhoH_ann_core_eps4[1][0][0])
fit_rhoH_ann_core_eps4 = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_eps4[0][0], M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_eps4_1p = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_eps4[0][0]+np.sqrt(coeffrhoH_ann_core_eps4[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_eps4_1m = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_eps4[0][0]-np.sqrt(coeffrhoH_ann_core_eps4[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]

coeffrhoH_ann_core_2eps = curve_fit(f=lambda r, rho_core: rhoH_ann_core(r, rho_core), xdata=mid_bins_sub_2eps, ydata=rho_bins_sub_2eps, p0=[rho_core_true], sigma=drho_bins_sub_2eps)
print "fitted rho_core_2eps = ", coeffrhoH_ann_core_2eps[0][0], " +- ", np.sqrt(coeffrhoH_ann_core_2eps[1][0][0])
fit_rhoH_ann_core_2eps = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_2eps[0][0], M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_2eps_1p = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_2eps[0][0]+np.sqrt(coeffrhoH_ann_core_2eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_2eps_1m = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_2eps[0][0]-np.sqrt(coeffrhoH_ann_core_2eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]

coeffrhoH_ann_core_4eps = curve_fit(f=lambda r, rho_core: rhoH_ann_core(r, rho_core), xdata=mid_bins_sub_4eps, ydata=rho_bins_sub_4eps, p0=[rho_core_true], sigma=drho_bins_sub_4eps)
print "fitted rho_core_4eps = ", coeffrhoH_ann_core_4eps[0][0], " +- ", np.sqrt(coeffrhoH_ann_core_4eps[1][0][0])
fit_rhoH_ann_core_4eps = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_4eps[0][0], M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_4eps_1p = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_4eps[0][0]+np.sqrt(coeffrhoH_ann_core_4eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_4eps_1m = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_4eps[0][0]-np.sqrt(coeffrhoH_ann_core_4eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]

coeffrhoH_ann_core_8eps = curve_fit(f=lambda r, rho_core: rhoH_ann_core(r, rho_core), xdata=mid_bins_sub_8eps, ydata=rho_bins_sub_8eps, p0=[rho_core_true], sigma=drho_bins_sub_8eps)
print "fitted rho_core_8eps = ", coeffrhoH_ann_core_8eps[0][0], " +- ", np.sqrt(coeffrhoH_ann_core_8eps[1][0][0])
fit_rhoH_ann_core_8eps = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_8eps[0][0], M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_8eps_1p = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_8eps[0][0]+np.sqrt(coeffrhoH_ann_core_8eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_8eps_1m = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core_8eps[0][0]-np.sqrt(coeffrhoH_ann_core_8eps[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]



rho_bins_sub_1eps_1p = [rho_bins_sub_1eps[j] + drho_bins_sub_1eps[j] for j in range(0, len(rho_bins_sub_1eps))]
rho_bins_sub_1eps_1m = [rho_bins_sub_1eps[j] - drho_bins_sub_1eps[j] for j in range(0, len(rho_bins_sub_1eps))]

rho_bins_sub_eps4_1p = [rho_bins_sub_eps4[j] + drho_bins_sub_eps4[j] for j in range(0, len(rho_bins_sub_eps4))]
rho_bins_sub_eps4_1m = [rho_bins_sub_eps4[j] - drho_bins_sub_eps4[j] for j in range(0, len(rho_bins_sub_eps4))]

rho_bins_sub_2eps_1p = [rho_bins_sub_2eps[j] + drho_bins_sub_2eps[j] for j in range(0, len(rho_bins_sub_2eps))]
rho_bins_sub_2eps_1m = [rho_bins_sub_2eps[j] - drho_bins_sub_2eps[j] for j in range(0, len(rho_bins_sub_2eps))]

rho_bins_sub_4eps_1p = [rho_bins_sub_4eps[j] + drho_bins_sub_4eps[j] for j in range(0, len(rho_bins_sub_4eps))]
rho_bins_sub_4eps_1m = [rho_bins_sub_4eps[j] - drho_bins_sub_4eps[j] for j in range(0, len(rho_bins_sub_4eps))]

rho_bins_sub_8eps_1p = [rho_bins_sub_8eps[j] + drho_bins_sub_8eps[j] for j in range(0, len(rho_bins_sub_8eps))]
rho_bins_sub_8eps_1m = [rho_bins_sub_8eps[j] - drho_bins_sub_8eps[j] for j in range(0, len(rho_bins_sub_8eps))]



##-- NORMALIZED quantities (no fitted quantities) --##
# rho -> rho/rhoH_cte
# r -> r/a

roa_min = r_min / a
roa_max = r_max / a

ptoa_bins = np.logspace(start=np.log10(roa_min), stop=np.log10(roa_max), num=1000)
rhoHorH_th = [el / rhoH_cte for el in rhoH_th]
fit_rhoHorH_ann = [el / rhoH_cte for el in fit_rhoH_ann]

mid_bins_sub_1epsN = [el / a for el in mid_bins_sub_1eps]
rho_bins_sub_1epsN = [el / rhoH_cte for el in rho_bins_sub_1eps]
rho_bins_sub_1epsN_1p = [el / rhoH_cte for el in rho_bins_sub_1eps_1p]
rho_bins_sub_1epsN_1m = [el / rhoH_cte for el in rho_bins_sub_1eps_1m]

mid_bins_sub_eps4N = [el / a for el in mid_bins_sub_eps4]
rho_bins_sub_eps4N = [el / rhoH_cte for el in rho_bins_sub_eps4]
rho_bins_sub_eps4N_1p = [el / rhoH_cte for el in rho_bins_sub_eps4_1p]
rho_bins_sub_eps4N_1m = [el / rhoH_cte for el in rho_bins_sub_eps4_1m]

mid_bins_sub_2epsN = [el / a for el in mid_bins_sub_2eps]
rho_bins_sub_2epsN = [el / rhoH_cte for el in rho_bins_sub_2eps]
rho_bins_sub_2epsN_1p = [el / rhoH_cte for el in rho_bins_sub_2eps_1p]
rho_bins_sub_2epsN_1m = [el / rhoH_cte for el in rho_bins_sub_2eps_1m]

mid_bins_sub_4epsN = [el / a for el in mid_bins_sub_4eps]
rho_bins_sub_4epsN = [el / rhoH_cte for el in rho_bins_sub_4eps]
rho_bins_sub_4epsN_1p = [el / rhoH_cte for el in rho_bins_sub_4eps_1p]
rho_bins_sub_4epsN_1m = [el / rhoH_cte for el in rho_bins_sub_4eps_1m]

mid_bins_sub_8epsN = [el / a for el in mid_bins_sub_8eps]
rho_bins_sub_8epsN = [el / rhoH_cte for el in rho_bins_sub_8eps]
rho_bins_sub_8epsN_1p = [el / rhoH_cte for el in rho_bins_sub_8eps_1p]
rho_bins_sub_8epsN_1m = [el / rhoH_cte for el in rho_bins_sub_8eps_1m]


##-- Plots --##
fig4 = plt.figure(num='rho_vs_r', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax4.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(1.71e0, 6.0e2) #(1.0e0, 2.9e2)
ax4.set_ylim(0.1e5, 3.0e8)
ax4.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0, label=r'original profile')
ax4.plot(pt_bins, fit_rhoH_ann, color ='darkviolet', linestyle = '-', lw=3.0, label=r'analytical')

##ax4.errorbar(mid_bins, rho_bins_eps4, xerr=0, yerr=drho_bins_eps4, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$h_A = \epsilon/4$')
##ax4.plot(pt_bins, fit_rhoH_ann_core_eps4, color ='cyan', linestyle = '--', lw=1.5)
##ax4.fill_between(pt_bins, fit_rhoH_ann_core_eps4_1m, fit_rhoH_ann_core_eps4_1p, color ='cyan', alpha=0.3)
ax4.plot(mid_bins_sub_eps4, rho_bins_sub_eps4, color ='cyan', linestyle = '-', lw=2.0, label=r'$h_A = \epsilon/4$')
ax4.fill_between(mid_bins_sub_eps4, rho_bins_sub_eps4_1m, rho_bins_sub_eps4_1p, color ='cyan', alpha=0.3)

#ax4.errorbar(mid_bins, rho_bins_1eps, xerr=0, yerr=drho_bins_1eps, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$h_A = \epsilon$')
#ax4.plot(pt_bins, fit_rhoH_ann_core_1eps, color ='blue', linestyle = '--', lw=1.5)
#ax4.fill_between(pt_bins, fit_rhoH_ann_core_1eps_1m, fit_rhoH_ann_core_1eps_1p, color ='blue', alpha=0.3)
ax4.plot(mid_bins_sub_1eps, rho_bins_sub_1eps, color ='blue', linestyle = '-', lw=2.0, label=r'$h_A = \epsilon$')
ax4.fill_between(mid_bins_sub_1eps, rho_bins_sub_1eps_1m, rho_bins_sub_1eps_1p, color ='blue', alpha=0.3)

#ax4.errorbar(mid_bins, rho_bins_2eps, xerr=0, yerr=drho_bins_2eps, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$h_A = 2\epsilon$')
#ax4.plot(pt_bins, fit_rhoH_ann_core_2eps, color ='green', linestyle = '--', lw=1.5)
#ax4.fill_between(pt_bins, fit_rhoH_ann_core_2eps_1m, fit_rhoH_ann_core_2eps_1p, color ='blue', alpha=0.3)
ax4.plot(mid_bins_sub_2eps, rho_bins_sub_2eps, color ='green', linestyle = '-', lw=2.0, label=r'$h_A = 2\epsilon$')
ax4.fill_between(mid_bins_sub_2eps, rho_bins_sub_2eps_1m, rho_bins_sub_2eps_1p, color ='green', alpha=0.3)

#ax4.errorbar(mid_bins, rho_bins_4eps, xerr=0, yerr=drho_bins_4eps, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$h_A = 4\epsilon$')
#ax4.plot(pt_bins, fit_rhoH_ann_core_4eps, color ='orange', linestyle = '--', lw=1.5)
#ax4.fill_between(pt_bins, fit_rhoH_ann_core_4eps_1m, fit_rhoH_ann_core_4eps_1p, color ='blue', alpha=0.3)
ax4.plot(mid_bins_sub_4eps, rho_bins_sub_4eps, color ='orange', linestyle = '-', lw=2.0, label=r'$h_A = 4\epsilon$')
ax4.fill_between(mid_bins_sub_4eps, rho_bins_sub_4eps_1m, rho_bins_sub_4eps_1p, color ='orange', alpha=0.3)

#ax4.errorbar(mid_bins, rho_bins_8eps, xerr=0, yerr=drho_bins_8eps, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$h_A = 8\epsilon$')
#ax4.plot(pt_bins, fit_rhoH_ann_core_8eps, color ='red', linestyle = '--', lw=1.5)
#ax4.fill_between(pt_bins, fit_rhoH_ann_core_8eps_1m, fit_rhoH_ann_core_8eps_1p, color ='blue', alpha=0.3)
ax4.plot(mid_bins_sub_8eps, rho_bins_sub_8eps, color ='red', linestyle = '-', lw=2.0, label=r'$h_A = 8\epsilon$')
ax4.fill_between(mid_bins_sub_8eps, rho_bins_sub_8eps_1m, rho_bins_sub_8eps_1p, color ='red', alpha=0.3)

ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.25*np.sqrt(eps), y=3.0e5, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=r'$\langle \sigma_{\rm ann} v \rangle /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob4 = offsetbox.AnchoredText(s=r'$\langle \sigma v \rangle_a /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()
#fig4.savefig('test2/figs/rho_vs_r_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



fig5 = plt.figure(num='rhorH_vs_roa', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r / a$', fontsize=20)
ax5.set_ylabel(r'$\rho (r) / \rho_H$', fontsize=20)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlim(1.71e0/a, 6.0e2/a) #(7.6e-3, 2.4)
ax5.set_ylim(0.1e5/rhoH_cte, 3.0e8/rhoH_cte) #(1.0e-2, 2.0e2)
ax5.plot(ptoa_bins, rhoHorH_th, color ='black', linestyle = '--', lw=2.0, label=r'original profile')
ax5.plot(ptoa_bins, fit_rhoHorH_ann, color ='darkviolet', linestyle = '-', lw=3.0, label=r'analytical')

ax5.plot(mid_bins_sub_eps4N, rho_bins_sub_eps4N, color ='cyan', linestyle = '-', lw=2.0, label=r'$h_A = \epsilon/4$')
ax5.fill_between(mid_bins_sub_eps4N, rho_bins_sub_eps4N_1m, rho_bins_sub_eps4N_1p, color ='cyan', alpha=0.3)

ax5.plot(mid_bins_sub_1epsN, rho_bins_sub_1epsN, color ='blue', linestyle = '-', lw=2.0, label=r'$h_A = \epsilon$')
ax5.fill_between(mid_bins_sub_1epsN, rho_bins_sub_1epsN_1m, rho_bins_sub_1epsN_1p, color ='blue', alpha=0.3)

ax5.plot(mid_bins_sub_2epsN, rho_bins_sub_2epsN, color ='green', linestyle = '-', lw=2.0, label=r'$h_A = 2\epsilon$')
ax5.fill_between(mid_bins_sub_2epsN, rho_bins_sub_2epsN_1m, rho_bins_sub_2epsN_1p, color ='green', alpha=0.3)

ax5.plot(mid_bins_sub_4epsN, rho_bins_sub_4epsN, color ='orange', linestyle = '-', lw=2.0, label=r'$h_A = 4\epsilon$')
ax5.fill_between(mid_bins_sub_4epsN, rho_bins_sub_4epsN_1m, rho_bins_sub_4epsN_1p, color ='orange', alpha=0.3)

ax5.plot(mid_bins_sub_8epsN, rho_bins_sub_8epsN, color ='red', linestyle = '-', lw=2.0, label=r'$h_A = 8\epsilon$')
ax5.fill_between(mid_bins_sub_8epsN, rho_bins_sub_8epsN_1m, rho_bins_sub_8epsN_1p, color ='red', alpha=0.3)

ax5.axvline(epsoa, color='gray', linestyle = '-.', lw=2.0)
ax5.text(x=epsoa + 2.0e-3, y=3.0e5/rhoH_cte, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.legend(loc='upper right', prop={'size': 18})
ob5 = offsetbox.AnchoredText(s=r'$\langle \sigma_{\rm ann} v \rangle /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob5 = offsetbox.AnchoredText(s=r'$\langle \sigma v \rangle_a /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
ob5_1 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()
#fig5.savefig('test2/figs/rhorH_vs_roa_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)




## THIRD PART ##
time_i = 0.0
time_02 = 0.999999979511
time_05 = 2.99999997765
time_07 = 4.99999997579
time_09 = 6.99999997392
time_13 = 9.99999999069

#-- these come from the code "annihilation_cmp_test2.py" --#
mid_bins_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 0) # same mid_bins for all of them!
dmid_bins_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 1) # same mid_bins for all of them!
rho_bins_i_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 2)
drho_bins_i_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 3)
rho_bins_02_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 4)
drho_bins_02_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 5)
rho_bins_05_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 6)
drho_bins_05_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 7)
rho_bins_07_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 8)
drho_bins_07_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 9)
rho_bins_09_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 10)
drho_bins_09_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 11)
rho_bins_13_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 12)
drho_bins_13_2 = np.loadtxt(fname='test2/annihilation/data/benchmark2_noption_annihilation.dat', delimiter='\t', usecols = 13)


# for fit cored Hernquist profile
def rhoH_ann(r, M=M*1.0e10, a=a, time=time_i) :
	rhoH_cte = M / (2.0 * np.pi * a**3)
	x = r / a
	rho_core = 1.0 / (sigmav_ann * time) /2.0 # (code units)
	rho_core *= 1.0e10 # since we want [Msun / kpc^3]
	if r == r_min :
		print "rho_core = ", rho_core, "--> time = ", time
	return rhoH_cte / (x * (1.0 + x)**3 + rhoH_cte / rho_core)

fit_rhoH_ann_02 = [rhoH_ann(r=el, M=M*1.0e10, a=a, time=time_02) for el in pt_bins]
fit_rhoH_ann_05 = [rhoH_ann(r=el, M=M*1.0e10, a=a, time=time_05) for el in pt_bins]
fit_rhoH_ann_07 = [rhoH_ann(r=el, M=M*1.0e10, a=a, time=time_07) for el in pt_bins]
fit_rhoH_ann_09 = [rhoH_ann(r=el, M=M*1.0e10, a=a, time=time_09) for el in pt_bins]
fit_rhoH_ann_13 = [rhoH_ann(r=el, M=M*1.0e10, a=a, time=time_13) for el in pt_bins]



# for fit cored Hernquist profile
def rhoCore(r, rcore, beta, M=M*1.0e10, a=a) :
	return M / (2.0 * np.pi) * a / (r**beta + rcore**beta)**(1.0/beta) / (r + a)**3 


mid_bins_sub_02 = []
rho_bins_sub_02 = []
drho_bins_sub_02 = []

mid_bins_sub_05 = []
rho_bins_sub_05 = []
drho_bins_sub_05 = []

mid_bins_sub_07 = []
rho_bins_sub_07 = []
drho_bins_sub_07 = []

mid_bins_sub_09 = []
rho_bins_sub_09 = []
drho_bins_sub_09 = []

mid_bins_sub_13 = []
rho_bins_sub_13 = []
drho_bins_sub_13 = []

for j in range(0, len(mid_bins)) :
	if rho_bins_02_2[j] >= rho_bins_02_2[-1] : # since rho(r) is monotonically decreasing function
		mid_bins_sub_02.append(mid_bins_2[j])
		rho_bins_sub_02.append(rho_bins_02_2[j])
		drho_bins_sub_02.append(drho_bins_02_2[j])

	if rho_bins_05_2[j] >= rho_bins_05_2[-1] :
		mid_bins_sub_05.append(mid_bins_2[j])
		rho_bins_sub_05.append(rho_bins_05_2[j])
		drho_bins_sub_05.append(drho_bins_05_2[j])

	if rho_bins_07_2[j] >= rho_bins_07_2[-1] :
		mid_bins_sub_07.append(mid_bins_2[j])
		rho_bins_sub_07.append(rho_bins_07_2[j])
		drho_bins_sub_07.append(drho_bins_07_2[j])

	if rho_bins_09_2[j] >= rho_bins_09_2[-1] :
		mid_bins_sub_09.append(mid_bins_2[j])
		rho_bins_sub_09.append(rho_bins_09_2[j])
		drho_bins_sub_09.append(drho_bins_09_2[j])

	if rho_bins_13_2[j] >= rho_bins_13_2[-1] :
		mid_bins_sub_13.append(mid_bins_2[j])
		rho_bins_sub_13.append(rho_bins_13_2[j])
		drho_bins_sub_13.append(drho_bins_13_2[j])

coeffrhoCore_02 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_sub_02, ydata=rho_bins_sub_02, p0=[0.1*a, 4.0], sigma=drho_bins_sub_02)
print "\nfitted [rcore, beta]_02 = ", coeffrhoCore_02[0]
fit_rho_02 = [rhoCore(el, coeffrhoCore_02[0][0], coeffrhoCore_02[0][1]) for el in pt_bins]

coeffrhoCore_05 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_sub_05, ydata=rho_bins_sub_05, p0=[0.1*a, 4.0], sigma=drho_bins_sub_05)
print "\nfitted [rcore, beta]_05 = ", coeffrhoCore_05[0]
fit_rho_05 = [rhoCore(el, coeffrhoCore_05[0][0], coeffrhoCore_05[0][1]) for el in pt_bins]

coeffrhoCore_07 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_sub_07, ydata=rho_bins_sub_07, p0=[0.1*a, 4.0], sigma=drho_bins_sub_07)
print "\nfitted [rcore, beta]_07 = ", coeffrhoCore_07[0]
fit_rho_07 = [rhoCore(el, coeffrhoCore_07[0][0], coeffrhoCore_07[0][1]) for el in pt_bins]

coeffrhoCore_09 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_sub_09, ydata=rho_bins_sub_09, p0=[0.1*a, 4.0], sigma=drho_bins_sub_09)
print "\nfitted [rcore, beta]_09 = ", coeffrhoCore_09[0]
fit_rho_09 = [rhoCore(el, coeffrhoCore_09[0][0], coeffrhoCore_09[0][1]) for el in pt_bins]

coeffrhoCore_13 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins_sub_13, ydata=rho_bins_sub_13, p0=[0.1*a, 4.0], sigma=drho_bins_sub_13)
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

fit_rhoH_ann_02N = [el / rhoH_cte for el in fit_rhoH_ann_02]
fit_rhoH_ann_05N = [el / rhoH_cte for el in fit_rhoH_ann_05]
fit_rhoH_ann_07N = [el / rhoH_cte for el in fit_rhoH_ann_07]
fit_rhoH_ann_09N = [el / rhoH_cte for el in fit_rhoH_ann_09]
fit_rhoH_ann_13N = [el / rhoH_cte for el in fit_rhoH_ann_13]

fit_rho_02N = [el / rhoH_cte for el in fit_rho_02]
fit_rho_05N = [el / rhoH_cte for el in fit_rho_05]
fit_rho_07N = [el / rhoH_cte for el in fit_rho_07]
fit_rho_09N = [el / rhoH_cte for el in fit_rho_09]
fit_rho_13N = [el / rhoH_cte for el in fit_rho_13]



fig7 = plt.figure(num='rho_vs_r_cmp_ann', figsize=(10, 7), dpi=100)
ax7 = fig7.add_subplot(111)
ax7.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax7.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax7.set_xscale('log')
ax7.set_yscale('log')
ax7.set_xlim(1.71e0, 6.0e2) #(1.0e0, 2.9e2)
ax7.set_ylim(0.1e5, 3.0e8)
ax7.errorbar(mid_bins_2, rho_bins_i_2, xerr=0, yerr=drho_bins_i_2, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax7.errorbar(mid_bins_2, rho_bins_02_2, xerr=0, yerr=drho_bins_02_2, c='cyan', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax7.errorbar(mid_bins_2, rho_bins_05_2, xerr=0, yerr=drho_bins_05_2, c='green', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax7.errorbar(mid_bins_2, rho_bins_07_2, xerr=0, yerr=drho_bins_07_2, c='yellow', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax7.errorbar(mid_bins_2, rho_bins_09_2, xerr=0, yerr=drho_bins_09_2, c='orange', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax7.errorbar(mid_bins_2, rho_bins_13_2, xerr=0, yerr=drho_bins_13_2, c='red', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax7.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax7.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax7.text(x=eps + 0.25*np.sqrt(eps), y=3.0e5, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax7.plot(pt_bins, fit_rhoH_ann_02, color ='cyan', linestyle = '-', lw=1.5)
ax7.plot(pt_bins, fit_rhoH_ann_05, color ='green', linestyle = '-', lw=1.5)
ax7.plot(pt_bins, fit_rhoH_ann_07, color ='yellow', linestyle = '-', lw=1.5)
ax7.plot(pt_bins, fit_rhoH_ann_09, color ='orange', linestyle = '-', lw=1.5)
ax7.plot(pt_bins, fit_rhoH_ann_13, color ='red', linestyle = '-', lw=1.5)

ax7.plot(pt_bins, fit_rho_02, color ='cyan', linestyle = ':', lw=2.0)
ax7.plot(pt_bins, fit_rho_05, color ='green', linestyle = ':', lw=2.0)
ax7.plot(pt_bins, fit_rho_07, color ='yellow', linestyle = ':', lw=2.0)
ax7.plot(pt_bins, fit_rho_09, color ='orange', linestyle = ':', lw=2.0)
ax7.plot(pt_bins, fit_rho_13, color ='red', linestyle = ':', lw=2.0)

ax7.grid(False)
ax7.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax7.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax7.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax7.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax7.legend(loc='upper right', prop={'size': 18})
ob7 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob7.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax7.add_artist(ob7)
ax7.legend(loc='upper right', prop={'size': 18})
ob7_1 = offsetbox.AnchoredText(s=r'$\langle \sigma_{\rm ann} v \rangle /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob7_1 = offsetbox.AnchoredText(s=r'$\langle \sigma v \rangle_a /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob7_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax7.add_artist(ob7_1)
fig7.tight_layout()
fig7.show()
#fig7.savefig('test2/figs/rho_vs_r_cmp_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



fig71 = plt.figure(num='rhorH_vs_roa_cmp_ann', figsize=(10, 7), dpi=100)
ax71 = fig71.add_subplot(111)
ax71.set_xlabel(r'$r / a$', fontsize=20)
ax71.set_ylabel(r'$\rho (r) /\rho_H$', fontsize=20)
ax71.set_xscale('log')
ax71.set_yscale('log')
ax71.set_xlim(1.71e0/a, 6.0e2/a) #(1.0e0, 2.9e2)
ax71.set_ylim(0.1e5/rhoH_cte, 3.0e8/rhoH_cte)
ax71.errorbar(mid_bins_2N, rho_bins_i_2N, xerr=0, yerr=drho_bins_i_2N, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax71.errorbar(mid_bins_2N, rho_bins_02_2N, xerr=0, yerr=drho_bins_02_2N, c='cyan', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax71.errorbar(mid_bins_2N, rho_bins_05_2N, xerr=0, yerr=drho_bins_05_2N, c='green', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax71.errorbar(mid_bins_2N, rho_bins_07_2N, xerr=0, yerr=drho_bins_07_2N, c='yellow', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax71.errorbar(mid_bins_2N, rho_bins_09_2N, xerr=0, yerr=drho_bins_09_2N, c='orange', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax71.errorbar(mid_bins_2N, rho_bins_13_2N, xerr=0, yerr=drho_bins_13_2N, c='red', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax71.plot(ptoa_bins, rhoHorH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax71.axvline(epsoa, color='gray', linestyle = '-.', lw=2.0)
ax71.text(x=epsoa + 2.0e-3, y=3.0e5/rhoH_cte, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax71.plot(ptoa_bins, fit_rhoH_ann_02N, color ='cyan', linestyle = '-', lw=1.5)
ax71.plot(ptoa_bins, fit_rhoH_ann_05N, color ='green', linestyle = '-', lw=1.5)
ax71.plot(ptoa_bins, fit_rhoH_ann_07N, color ='yellow', linestyle = '-', lw=1.5)
ax71.plot(ptoa_bins, fit_rhoH_ann_09N, color ='orange', linestyle = '-', lw=1.5)
ax71.plot(ptoa_bins, fit_rhoH_ann_13N, color ='red', linestyle = '-', lw=1.5)

ax71.plot(ptoa_bins, fit_rho_02N, color ='cyan', linestyle = ':', lw=2.0)
ax71.plot(ptoa_bins, fit_rho_05N, color ='green', linestyle = ':', lw=2.0)
ax71.plot(ptoa_bins, fit_rho_07N, color ='yellow', linestyle = ':', lw=2.0)
ax71.plot(ptoa_bins, fit_rho_09N, color ='orange', linestyle = ':', lw=2.0)
ax71.plot(ptoa_bins, fit_rho_13N, color ='red', linestyle = ':', lw=2.0)

ax71.grid(False)
ax71.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax71.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax71.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax71.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax71.legend(loc='upper right', prop={'size': 18})
ob71 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob71.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax71.add_artist(ob71)
ax71.legend(loc='upper right', prop={'size': 18})
ob71_1 = offsetbox.AnchoredText(s=r'$\langle \sigma_{\rm ann} v \rangle /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob71_1 = offsetbox.AnchoredText(s=r'$\langle \sigma v \rangle_a /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob71_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax71.add_artist(ob71_1)
fig71.tight_layout()
fig71.show()
#fig71.savefig('test2/figs/rhorH_vs_roa_cmp_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)






raw_input()
