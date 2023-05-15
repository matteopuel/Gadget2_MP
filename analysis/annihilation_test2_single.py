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

def MrH(r, M, a) :
	return M * r**3 / (r + a)**2


# Hernquist and simulation parameters in code units!
##---------# (used for test 2) #---------#
#infile_ini = "test2/hernquist_test_v1"
#infile_ini = "test2/stability/benchmark/out/snp_013" # almost identical outcome of the original one at t = 0
#M = 1.0e5 # * 1e10 Msun (total mass -> galaxy cluster)
#a = 1.0e3 # kpc (scale radius)
#eps = 12.0 # kpc
#infile = "test2/annihilation/benchmark_noption/out/snp_013"


#-------# (used for stability) #-------#
#infile_ini = "test2/hernquist_test2_v2"
infile_ini = "test2/stability/benchmark2/out/snp_013"
M = 1.0e4 # * 1e10 Msun
a = 225.0 # kpc
eps = 4.4 # kpc

#fn = "benchmark2_noption" # 1eps
fn = "benchmark2_eps4" # eps/4
#fn = "benchmark2_2eps"
#fn = "benchmark2_4eps"
#fn = "benchmark2_8eps"
#fn = "benchmark2_hcs" # no!
#fn = "benchmark2_newANN" # no! implementation as Iwanus et al.

infile = "test2/annihilation/" + fn + "/out/snp_013"
str_out = "test2/annihilation/data/" + fn + ".dat"


##-------# (used for checks) #-------#
#infile_ini = "test2/hernquist_test3_v1"
#M = 3.0 # * 1e10 Msun
#a = 10.0 # kpc
#eps = 0.3 # kpc
#infile = "test2/annihilation/benchmark3_noption/out/snp_013"

massopt = False


# dynamical time = orbital time at r = a [https://iopscience.iop.org/article/10.1086/317149/pdf]
tdyn = 4.0 * np.pi * np.sqrt(a**3 / (G * M)) # (code units)
print "tdyn = ", tdyn, " (code units)"
print "     = ", tdyn * (UnitTime_in_s / Gyr), " Gyr\n" 

tmassloss = 1.0 / sigmav_ann / rhoH(r=eps, M=M, a=a) # rouhgly minimum radius
print "tmassloss (r = eps) = ", tmassloss, " (code units)"
print "                    = ", tmassloss * (UnitTime_in_s / Gyr), " Gyr\n"



##-- Set the IC file --##
Ntot = readGadget1.readHeader(filename=infile, strname='npart')
Mtot = readGadget1.readHeader(filename=infile, strname='mass')
time = readGadget1.readHeader(filename=infile, strname='time')

print "Ntot = ", Ntot
print "Mtot = ", Mtot # (code units)
print "time = ", time, "\n" # (code units)

NDM = Ntot[1]


Xmax = 0.99
M /= Xmax # the mass is a bit larger if we want that M(<r_max) = mass we want


##-- Read the particle properties --##
PPosDM_ini, _, _, PMassDM1_ini = readGadget1.readSnapshot(filename=infile_ini, ptype='dm', strname='pos', full=True, mass=False)
PPosDM, _, _, PMassDM1 = readGadget1.readSnapshot(filename=infile, ptype='dm', strname='pos', full=True, mass=massopt)

PMassDM_ini = [el*1.0e10 for el in PMassDM1_ini]
PMassDM = [el*1.0e10 for el in PMassDM1]


##-- Velocity distribution --##
# (Maxwell-Boltzmann? NO, maybe because f(0.5 * v**2 - G * M / (r + a)) is more complicated)
x, y, z = getCoord(vec=PPosDM)
pos_r = getRad(x=x, y=y, z=z)

x_ini, y_ini, z_ini = getCoord(vec=PPosDM_ini)
pos_r_ini = getRad(x=x_ini, y=y_ini, z=z_ini)


numbins = 30


r_min = np.around(a=min(pos_r), decimals=1)
r_max = np.around(a=max(pos_r), decimals=0) + 1.0
print "\nr_min = ", r_min, " kpc"
print "r_max = ", r_max, " kpc"

r_min = 1.0 # kpc
#r_min = 0.1
r_max = 100.0 * a
if r_min == 0.0 :
	r_min = 0.01


bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # left edges


Npart_bins_ini = [0 for j in range(0, len(bins)-1)]
Mp_bins_ini = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_ini)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_ini[i] >= bins[j] and pos_r_ini[i] < bins[j+1] :
			Npart_bins_ini[j] += 1
			Mp_bins_ini[j].append(PMassDM_ini[i])
			break

dNpart_bins_ini = [np.sqrt(el) for el in Npart_bins_ini]

Npart_bins = [0 for j in range(0, len(bins)-1)]
Mp_bins = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r)) :	
	for j in range(0, len(bins)-1) :
		if pos_r[i] >= bins[j] and pos_r[i] < bins[j+1] :
			Npart_bins[j] += 1
			Mp_bins[j].append(PMassDM[i])
			break

dNpart_bins = [np.sqrt(el) for el in Npart_bins]


mid_bins = []
dmid_bins = []

for j in range(0, len(bins)-1) :
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)


rho_bins_ini = []
drho_bins_ini = []
Menc_bins_ini = []
dMenc_bins_ini = []
menc_ini = 0
dmenc2_ini = 0

for j in range(0, len(bins)-1) :
	if (Npart_bins_ini[j] != 0) :
		mass_j = sum(Mp_bins_ini[j])
		dmass_j = np.sqrt(Npart_bins_ini[j]) * (mass_j / Npart_bins_ini[j]) # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_ini.append(mass_j / vol_j)
		drho_bins_ini.append(dmass_j / vol_j)
		menc_ini += mass_j
		dmenc2_ini += dmass_j**2
	else :
		rho_bins_ini.append(0.0)
		drho_bins_ini.append(0.0)
		menc_ini += 0.0
		dmenc2_ini += 0.0
	
	Menc_bins_ini.append(menc_ini)
	dMenc_bins_ini.append(np.sqrt(dmenc2_ini))


rho_bins = []
drho_bins = []
Menc_bins = []
dMenc_bins = []
menc = 0
dmenc2 = 0

for j in range(0, len(bins)-1) :
	if (Npart_bins[j] != 0) :
		mass_j = sum(Mp_bins[j])
		dmass_j = np.sqrt(Npart_bins[j]) * (mass_j / Npart_bins[j]) # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins.append(mass_j / vol_j)
		drho_bins.append(dmass_j / vol_j)
		menc += mass_j
		dmenc2 += dmass_j**2
	else :
		rho_bins.append(0.0)
		drho_bins.append(0.0) 
		menc += 0.0
		dmenc2 += 0.0


	Menc_bins.append(menc)
	dMenc_bins.append(np.sqrt(dmenc2))


Mr_bins_ini = []
dMr_bins_ini = []
Mr_bins = []
dMr_bins = []
for j in range(0, len(bins)-1) :
	mr_j_ini = Menc_bins_ini[j] * mid_bins[j]
	dmr_j_ini = np.sqrt((mid_bins[j] * dMenc_bins_ini[j])**2 + (Menc_bins_ini[j] * dmid_bins[j])**2)
	Mr_bins_ini.append(mr_j_ini)
	dMr_bins_ini.append(dmr_j_ini)

	mr_j = Menc_bins[j] * mid_bins[j]
	dmr_j = np.sqrt((mid_bins[j] * dMenc_bins[j])**2 + (Menc_bins[j] * dmid_bins[j])**2)
	Mr_bins.append(mr_j)
	dMr_bins.append(dmr_j)



## save the information for each value of h_A ##
file_out = open(str_out, "w+")
#file_out.write("mid_bins\trho_bins\tdmid_bins\tdrho_bins\n")
for i in range(0, len(rho_bins)) :
	file_out.write("%.5f\t%.5f\t%.5f\t%.5f\n" %(mid_bins[i], rho_bins[i], dmid_bins[i], drho_bins[i]))
file_out.close()



pt_bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=1000)
rhoH_th = [rhoH(r=el, M=M*1.0e10, a=a) for el in pt_bins]
MrH_th = [MrH(r=el, M=M*1.0e10, a=a) for el in pt_bins]


# for annihilation-cored Hernquist profile
def rhoH_ann(r, M=M*1.0e10, a=a, time=0.0) :
	rhoH_cte = M / (2.0 * np.pi * a**3)
	x = r / a
	rho_core = 1.0 / (sigmav_ann * time) / 2.0 # (code units) --> 2 units of mass disappears, not just one!
	rho_core *= 1.0e10 # since we want [Msun / kpc^3]
	if r == r_min :
		print "\nrho_core = ", rho_core
	return rhoH_cte / (x * (1.0 + x)**3 + rhoH_cte / rho_core)

fit_rhoH_ann = [rhoH_ann(r=el, M=M*1.0e10, a=a, time=time) for el in pt_bins]
rho_core_true = 1.0 / (sigmav_ann * time) * 1.0e10 # Msun / kpc^3


def rhoH_ann_core(r, rho_core, M=M*1.0e10, a=a) :
	rhoH_cte = M / (2.0 * np.pi * a**3)
	x = r / a
	return rhoH_cte / (x * (1.0 + x)**3 + rhoH_cte / rho_core)

mid_bins_sub = []
rho_bins_sub = []
drho_bins_sub = []
for j in range(0, len(mid_bins)) :
	if rho_bins[j] >= rho_bins[-1] : # since rho(r) is monotonically decreasing function
		mid_bins_sub.append(mid_bins[j])
		rho_bins_sub.append(rho_bins[j])
		drho_bins_sub.append(drho_bins[j])

coeffrhoH_ann_core = curve_fit(f=lambda r, rho_core: rhoH_ann_core(r, rho_core), xdata=mid_bins_sub, ydata=rho_bins_sub, p0=[rho_core_true], sigma=drho_bins_sub)
print "fitted rho_core = ", coeffrhoH_ann_core[0][0], " +- ", np.sqrt(coeffrhoH_ann_core[1][0][0])
fit_rhoH_ann_core = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core[0][0], M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_1p = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core[0][0]+np.sqrt(coeffrhoH_ann_core[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]
fit_rhoH_ann_core_1m = [rhoH_ann_core(r=el, rho_core=coeffrhoH_ann_core[0][0]-np.sqrt(coeffrhoH_ann_core[1][0][0]), M=M*1.0e10, a=a) for el in pt_bins]



r_max = float(r_max / 10.0)


##-- Plots --##
fig4 = plt.figure(num='rho_vs_r', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax4.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(r_min, r_max)
ax4.set_ylim(1.0e0, 1.0e9)
#ax4.set_ylim(1.0e2, 1.0e9)
ax4.errorbar(mid_bins, rho_bins, xerr=0, yerr=drho_bins, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$h_A = \epsilon$')
ax4.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0, label=r'original profile')
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.25*np.sqrt(eps), y=5.0e3, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.plot(pt_bins, fit_rhoH_ann, color ='red', linestyle = '-', lw=2.0, label=r'analytical')
ax4.plot(pt_bins, fit_rhoH_ann_core, color ='orange', linestyle = '--', lw=1.5, label=r'fit')
ax4.fill_between(pt_bins, fit_rhoH_ann_core_1m, fit_rhoH_ann_core_1p, color ='orange', alpha=0.3)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=r'$t = $ ' + str(np.around(a=time, decimals=1)) + r' Gyr', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()
#fig4.savefig('test2/figs/rho_vs_r_ann.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig4.savefig('test2/figs/rho_vs_r_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


fig5 = plt.figure(num='Mr_vs_r', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$M(r) \cdot r$ [M$_{\odot}$ kpc]', fontsize=20)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlim(r_min, r_max)
ax5.set_ylim(5.0e8, 1.0e21)
ax5.errorbar(mid_bins, Mr_bins_ini, xerr=dmid_bins, yerr=dMr_bins_ini, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'CDM')
ax5.errorbar(mid_bins, Mr_bins, xerr=dmid_bins, yerr=dMr_bins, c='green', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'annihilation')
ax5.plot(pt_bins, MrH_th, color ='red', linestyle = '-', lw=2.0, label=r'analytical')
ax5.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax5.text(x=eps + 0.25*np.sqrt(eps), y=1.0e17, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='all', numticks=15))
ax5.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax5.legend(loc='upper right', prop={'size': 18})
ob5 = offsetbox.AnchoredText(s=r'$t = $ ' + str(np.around(a=time, decimals=1)) + r' Gyr', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
ob5_1 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='upper left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()
#fig5.savefig('test2/figs/Mr_vs_r_ann.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig5.savefig('test2/figs/Mr_vs_r_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



raw_input()
