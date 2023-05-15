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


def rhoH(r, M, a) :
	return M / (2.0 * np.pi) * (a / r) / (r + a)**3


def veldispH(r, M, a) :
	fH = 12.0 * r * (r + a)**3 / a**4 * np.log((r + a)/r) - r/(r + a) * (25.0 + 52.0 * (r / a) + 42.0 * (r / a)**2 + 12.0 * (r / a)**3)
	return np.sqrt(G * M / (12.0 * a) * fH)


def Phi(r, M, a) :
	return - G * M / (r + a)

def vesc(r, M, a) :
	return np.sqrt(- 2.0 * Phi(r, M, a))


# Hernquist and simulation parameters in code units!
#------# (real simulation: DDO 154) #-------#
eps = 0.3 # kpc

## Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
#infile = "test3/hernquist/hernquist_v1_S"
#M = 2.3 # * 1e10 Msun
#a = 8.982285853787648 # kpc
infile = "test3/hernquist/hernquist_v1_S1"
M = 1.226850076349031 # * 1e10 Msun
a = 6.187578545092555 # kpc

infile = "test3/hernquist/CDM/DDO154/out/snp_013"
#infile = "test3/hernquist/vector/DDO154_benchmark/out/snp_013"
#infile = "test3/hernquist/scalar/DDO154_benchmark/out/snp_013"



##-- Set the IC file --##
Ntot = readGadget1.readHeader(filename=infile, strname='npart')
Mtot = readGadget1.readHeader(filename=infile, strname='mass')
time = readGadget1.readHeader(filename=infile, strname='time')

print "Ntot = ", Ntot
print "Mtot = ", Mtot
print "time = ", time, "\n"

mp = Mtot[1] # 1e10 Msun
NDM = Ntot[1]


Xmax = 0.99
M /= Xmax # the mass is a bit larger if we want that M(<r_max) = mass we want


##-- Read the particle properties --##
PPosDM, PVelDM, PIdDM, _ = readGadget1.readSnapshot(filename=infile, ptype='dm', strname='full', full=True, mass=False)


##-- Velocity distribution --##
# (Maxwell-Boltzmann? NO, maybe because f(0.5 * v**2 - G * M / (r + a)) is more complicated)
x, y, z = getCoord(vec=PPosDM)
vel_x, vel_y, vel_z = getCoord(vec=PVelDM)


# compute CoM quantities
x_com = y_com = z_com = 0
vx_com = vy_com = vz_com = 0

for i in range(0, NDM) :
	x_com += (x[i] * mp / Xmax / M) # Xmax dependence should be cancel out
	y_com += (y[i] * mp / Xmax / M)
	z_com += (z[i] * mp / Xmax / M)
	vx_com += (vel_x[i] * mp / Xmax / M)
	vy_com += (vel_y[i] * mp / Xmax / M)
	vz_com += (vel_z[i] * mp / Xmax / M)

print "r_com = ", [x_com, y_com, z_com] # difference between this value and that in hernquist_test.py is between float (former) and dobule (latter)
print "v_com = ", [vx_com, vy_com, vz_com]


pos_r = getRad(x=x, y=y, z=z)
vel_tot = getRad(x=vel_x, y=vel_y, z=vel_z)


# velocity decomposition in spherical coordinates
phi = [np.arctan(y[i] / x[i]) for i in range(0, len(pos_r))] 
theta = [np.arccos(z[i] / pos_r[i]) for i in range(0, len(pos_r))]

# Andrew's method
#vel_r = [vel_x[i]*x[i]/pos_r[i] + vel_y[i]*y[i]/pos_r[i] + vel_z[i]*z[i]/pos_r[i] for i in range(0, len(vel_tot))]

# my method
vel_r = [vel_x[i]*np.sin(theta[i])*np.cos(phi[i]) + vel_y[i]*np.sin(theta[i])*np.sin(phi[i]) + vel_z[i]*np.cos(theta[i]) for i in range(0, len(vel_tot))]
vel_theta = [vel_x[i]*np.cos(theta[i])*np.cos(phi[i]) + vel_y[i]*np.cos(theta[i])*np.sin(phi[i]) - vel_z[i]*np.sin(theta[i]) for i in range(0, len(vel_tot))]
vel_phi = [-vel_x[i]*np.sin(phi[i]) + vel_y[i]*np.cos(phi[i]) for i in range(0, len(vel_tot))] 



## Plot v(r) vs r (similar to what we will do for density rho(r) vs r, radial shells)
numbins = 30
mp *= 1.0e10 # Msun

r_min = np.around(a=min(pos_r), decimals=1)
r_max = np.around(a=max(pos_r), decimals=0) + 1.0
print "\nr_min = ", r_min, " kpc"
print "r_max = ", r_max, " kpc"

r_max = 100.0 * a
if r_min == 0.0 :
	r_min = 0.01
else :
	r_min = 1.0 # kpc


bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # left edges


Npart_bins = [0 for j in range(0, len(bins)-1)]
vel_bins = [[] for j in range(0, len(bins)-1)]
pos_bins = [[] for j in range(0, len(bins)-1)]

vel_r_bins = [[] for j in range(0, len(bins)-1)]
vel_theta_bins = [[] for j in range(0, len(bins)-1)]
vel_phi_bins = [[] for j in range(0, len(bins)-1)]

vel_x_bins = [[] for j in range(0, len(bins)-1)]
vel_y_bins = [[] for j in range(0, len(bins)-1)]
vel_z_bins = [[] for j in range(0, len(bins)-1)]

for i in range(0, len(pos_r)) :	
	for j in range(0, len(bins)-1) :
		if pos_r[i] >= bins[j] and pos_r[i] < bins[j+1] :
			Npart_bins[j] += 1
			vel_bins[j].append(vel_r[i])
			pos_bins[j].append(pos_r[i])

			vel_r_bins[j].append(vel_r[i])
			vel_theta_bins[j].append(vel_theta[i])
			vel_phi_bins[j].append(vel_phi[i])

			vel_x_bins[j].append(vel_x[i])
			vel_y_bins[j].append(vel_y[i])
			vel_z_bins[j].append(vel_z[i])
			break


dNpart_bins = [np.sqrt(el) for el in Npart_bins]


mid_bins = []
dmid_bins = []
vel_mean_bins = []
dvel_mean_bins = []
for j in range(0, len(bins)-1) :
	vel_sum_j = sum(vel_bins[j]) # len(vel_bins[j]) == Npart_bins[j]
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)
	if (Npart_bins[j] != 0) :
		vel_mean_bins.append(vel_sum_j/Npart_bins[j])
		dvel_mean_bins.append(vel_sum_j/(Npart_bins[j]**1.5))
	else :
		vel_mean_bins.append(0.0)
		dvel_mean_bins.append(0.0)


#for i in range(0, len(mid_bins)) :
#	print "i = ", i, "mid_bins = ", mid_bins[i]
#sys.exit() # i = 15--25 ranges for r = 3--200 kpc


vel_std_bins = []
dvel_std_bins = []
for j in range(0, len(vel_bins)) : # len(vel_bins) == len(bins)-1
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins[j])) :
		diff2_j += (vel_bins[j][i] - vel_mean_bins[j])**2
		diff_j += (vel_bins[j][i] - vel_mean_bins[j])
	if Npart_bins[j] != 0 :
		vel_std_bins.append(np.sqrt(diff2_j / Npart_bins[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins[j] * dvel_mean_bins[j])**2 + (diff2_j / Npart_bins[j]**2 * dNpart_bins[j])**2
			dvel_std_bins.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins[j])))
		else :
			dvel_std_bins.append(0.0)
	else :
		vel_std_bins.append(0.0)
		dvel_std_bins.append(0.0)


veldispH_th = [veldispH(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)] # G*M has right units #/np.sqrt(3.0) would work!
vescH = [vesc(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)]

dvel_std_bins_1p = [(vel_std_bins[j] + dvel_std_bins[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_1m = [(vel_std_bins[j] - dvel_std_bins[j]) for j in range(0, len(bins)-1)]

pos_bins_tot = list(chain.from_iterable(pos_bins))
vel_bins_tot = list(chain.from_iterable(vel_bins))



## unit conversion factors: 10^10 Msun/kpc^3 -> GeV/cm^3 ##
pc = 3.08567758149e16 # m
kpc = pc * 1.0e3 / 1.0e-2 # cm  
c = 299792458.0 # m/s
eV = 1.602176634e-19 # J
GeV = eV * 1.0e9
GeVoc2 = GeV / c**2 # kg
Msun = 1.98841e30 # kg
Msun /= GeVoc2 # GeV/c^2
convfac = Msun / (kpc**3) # GeV/cm^3


r_max = float(r_max / 10.0)


##-- Plot velocity distribution --##
fig0 = plt.figure(num='vel_tot', figsize=(10, 7), dpi=100) # --> check isotropy: True!
ax0 = fig0.add_subplot(111)
ax0.set_xlabel(r'$v_i$ [km/s] ', fontsize=20)
ax0.set_ylabel(r'$f(v_i)$', fontsize=20)
ax0.set_xscale('linear')
ax0.set_yscale('linear')
ax0.set_xlim(-100.0, 100.0)
ax0.set_ylim(0, 2.25e-2)
n01, bins01, _ = ax0.hist(vel_x, bins=100, range=None, histtype='bar', density=True, edgecolor='blue', color='blue', alpha=0.5, label=r'$v_x$')
n02, bins02, _ = ax0.hist(vel_y, bins=100, range=None, histtype='bar', density=True, edgecolor='red', color='red', alpha=0.5, label=r'$v_y$')
n03, bins03, _ = ax0.hist(vel_z, bins=100, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'$v_z$')
ax0.grid(False)
ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax0.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
ax0.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
ax0.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.005))
ax0.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.0025))
ax0.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax0.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax0.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax0.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax0.legend(loc='upper right', prop={'size': 18})
fig0.tight_layout()
fig0.show()


fig1 = plt.figure(num='f_vr', figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r'$v_r$ [km/s] ', fontsize=20)
ax1.set_ylabel(r'$f(v_r)$', fontsize=20)
ax1.set_xscale('linear')
ax1.set_yscale('linear')
ax1.set_xlim(-2500.0, 2500.0)
n4, bins4, _ = ax1.hist(vel_r, bins=100, range=None, histtype='bar', density=False, edgecolor='blue', color='blue', alpha=0.5)
ax1.grid(False)
ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
#ax1.legend(loc='upper right', prop={'size': 18})
fig1.tight_layout()
fig1.show()


fig2 = plt.figure(num='v_vs_r', figsize=(10, 7), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax2.set_ylabel(r'$v_r (r)$ [km/s]', fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(r_min, r_max)
#ax2.set_ylim(1.0, 5.0e3)
ax2.errorbar(mid_bins, vel_mean_bins, xerr=dmid_bins, yerr=dvel_mean_bins, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'mean data')
ax2.plot(mid_bins, vescH, color ='green', linestyle = '-.', lw=2.0, label=r'$v_{\rm esc}$')
ax2.scatter(pos_bins_tot, vel_bins_tot, c='cyan', marker='o', alpha=0.5, label=r'data')
ax2.grid(False)
ax2.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax2.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax2.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax2.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax2.legend(loc='upper right', prop={'size': 18})
fig2.tight_layout()
fig2.show()


fig3 = plt.figure(num='stdv_vs_r', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax3.set_ylabel(r'$\sigma_{v_r} (r)$ [km/s]', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(r_min, r_max)
#ax3.set_ylim(1.0e1, 5.0e3)
ax3.errorbar(mid_bins, vel_std_bins, xerr=dmid_bins, yerr=dvel_std_bins, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'data')
ax3.plot(mid_bins, veldispH_th, color ='red', linestyle = '--', lw=2.0, label=r'analytical')
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='upper right', prop={'size': 18})
ob3 = offsetbox.AnchoredText(s=r'$t = $ ' + str(np.around(a=time, decimals=1)) + r' Gyr', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
ob3_1 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3_1)
fig3.tight_layout()
fig3.show()


fig5 = plt.figure(num='Npart_vs_r', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$N_{\rm part}$', fontsize=20)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlim(r_min, r_max)
ax5.set_ylim(1.0, NDM)
ax5.errorbar(mid_bins, Npart_bins, xerr=dmid_bins, yerr=dNpart_bins, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'data')
ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.legend(loc='upper left', prop={'size': 18})
fig5.tight_layout()
fig5.show()



## analysis of velocity components in different radial bins ##
def plot_vel_bins(i, xmin=-100.0, xmax=100.0, ymin=0, ymax=2.25e-2, nbins=100) :
	mean_vel_r_bins_i = np.mean(vel_r_bins[i])
	std_vel_r_bins_i = np.std(vel_r_bins[i])
	#mean_vel_x_bins_i = np.mean(vel_x_bins[i])
	#std_vel_x_bins_i = np.std(vel_x_bins[i])

	#mean_vel_theta_bins_i = np.mean(vel_theta_bins[i])
	#std_vel_theta_bins_i = np.std(vel_theta_bins[i])
	mean_vel_y_bins_i = np.mean(vel_y_bins[i])
	std_vel_y_bins_i = np.std(vel_y_bins[i])

	mean_vel_phi_bins_i = np.mean(vel_phi_bins[i])
	std_vel_phi_bins_i = np.std(vel_phi_bins[i])
	#mean_vel_z_bins_i = np.mean(vel_z_bins[i])
	#std_vel_z_bins_i = np.std(vel_z_bins[i])

	fig = plt.figure(num='vel_bins_%d' % i, figsize=(10, 7), dpi=100)
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$v_i$ [km/s] ', fontsize=20)
	ax.set_ylabel(r'$f(v_i)$', fontsize=20)
	ax.set_xscale('linear')
	ax.set_yscale('linear')
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.hist(vel_r_bins[i], bins=nbins, range=None, histtype='bar', density=True, edgecolor='blue', color='blue', alpha=0.5, label=r'$v_r$' + ', m = %.2f, std = %.2f' % (mean_vel_r_bins_i, std_vel_r_bins_i))
	#ax.hist(vel_x_bins[i], bins=nbins, range=None, histtype='bar', density=True, edgecolor='blue', color='red', alpha=0.5, label=r'$v_x$' + ', m = %.2f, std = %.2f' % (mean_vel_x_bins_i, std_vel_x_bins_i))
	#ax.hist(vel_theta_bins[i], bins=nbins, range=None, histtype='bar', density=True, edgecolor='red', color='red', alpha=0.5, label=r'$v_{\theta}$ + ', m = %.2f, std = %.2f' % (mean_vel_theta_bins_i, std_vel_theta_bins_i)')
	ax.hist(vel_y_bins[i], bins=nbins, range=None, histtype='bar', density=True, edgecolor='red', color='red', alpha=0.5, label=r'$v_y$' + ', m = %.2f, std = %.2f' % (mean_vel_y_bins_i, std_vel_y_bins_i))
	ax.hist(vel_phi_bins[i], bins=nbins, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'$v_{\phi}$' + ', m = %.2f, std = %.2f' % (mean_vel_phi_bins_i, std_vel_phi_bins_i))
	#ax.hist(vel_z_bins[i], bins=nbins, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'$v_z$' + ', m = %.2f, std = %.2f' % (mean_vel_z_bins_i, std_vel_z_bins_i))
	ax.grid(False)
	ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
	ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
	ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
	ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.005))
	ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.0025))
	ax.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
	ax.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
	ax.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
	ax.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
	ax.legend(loc='upper right', prop={'size': 18})
	ob = offsetbox.AnchoredText(s=r'r $\in$' + ' [%.2f, %.2f] kpc' % (bins[i], bins[i+1]) + '\nN = %d particles' % len(vel_r_bins[i]), loc='upper left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
	ob.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
	ax.add_artist(ob)
	fig.tight_layout()
	fig.show()

	return

for j in range(6, len(mid_bins)) :
	plot_vel_bins(j, xmin=-100.0, xmax=100.0, ymin=0, ymax=2.25e-2, nbins=100)



raw_input()
