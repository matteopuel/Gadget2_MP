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

from scipy.special import spence

import readGadget1 


from tqdm import tqdm 



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

#print "rhocrit = ", rhocrit
#sys.exit()

vbar = 200.0

def M200(r200) :
	return 4.0 * np.pi / 3.0 * r200**3 * vbar * rhocrit 

def R200(m200) :
	r3 = 3.0 * m200 / (4.0 * np.pi * vbar * rhocrit)
	return r3**(1.0/3.0)


def C1(c, Vv) :
	return Vv**2 * gc(c)

def Phi(s, c, Vv) :
	# s = (r / rv), but c = rv / rs
	x = c * s
	if x == 0 :
		return -C1(c, Vv) * c
	else :
		return -C1(c, Vv) * c * np.log(1.0 + x) / x

def gc(c) :
	tmp = np.log(1.0 + c) - c / (1.0 + c)
	return 1.0 / tmp

def Ms(s, c, Mv) :
	x = c * s # = r / rs
	return Mv * gc(c) * (np.log(1.0 + x) - x / (1.0 + x))

def C2(c, Mv, rv) :
	return Mv * c**2 * gc(c) / (4.0 * np.pi * rv**3)

def rhos(s, c, Mv, rv) :
	x = c * s
	return C2(c, Mv, rv) * c / (x * (1.0 + x)**2)

def rhoNFW(r, rhoss, rs) :
	return rhoss / (r / rs) / (1.0 + r / rs)**2


def vesc(s, c, Vv) :
	return np.sqrt(- 2.0 * Phi(s, c, Vv))


def veldisps(s, c, Vv) :
	x = c * s
	arg = np.pi**2 - np.log(x) - 1.0 / x - 1.0 / (1.0 + x)**2 - 6.0 / (1.0 + x) + (1.0 + 1.0 / x**2 - 4.0 / x - 2.0 / (1.0 + x)) * np.log(1.0 + x) + 3.0 * (np.log(1.0 + x))**2 + 6.0 * spence(1.0 + x)
	tmp = 0.5 * c**2 * gc(c) * s * (1.0 + x)**2 * arg
	if tmp < 0 :
		tmp = 0
	return Vv * np.sqrt(tmp)



# NFW and simulation parameters in code units!
##---------# (DDO 154 dSph galaxy) #---------#
#infile = "test3/nfw_test3_dsph_v2_4"
#infile = "test3/stability/DDO154_benchmark/out4/snp_013"

#infile = "test3/nfw_test3_dsph_v1"
##infile = "test3/stability/DDO154_benchmark/out/snp_013"

#infile = "test3/nfw_test3_dsph_v1_try"
#infile = "test3/nfw_test3_dsph_v2_try"
infile = "test3/stability/DDO154_benchmark/out2_try/snp_013"

#infile = "test3/nfw_test3_dsph_v2"
Mv1 = 2.3 # * 1e10 Msun (total mass -> dSph galaxy)
rv1 = R200(Mv1) # kpc
rhos1 = 1.5e-3 # * 1e10 Msun/kpc^3
rs1 = 3.4 # kpc
Vv1 = 49.0 # km/s
c1 = 12.2
eps = 0.3 # kpc


###---------# (A2537 galaxy cluster) #---------#
#outfile = "test3/nfw_test3_cluster_v1"
##outfile = "test3/nfw_test3_cluster_v2"
#rv1 = 2050.0 # kpc
#Mv1 = M200(rv1) # * 1e10 Msun (total mass -> galaxy cluster)
#rhos1 = 1.3e-4 # * 1e10 Msun/kpc^3
#rs1 = 442 # kpc
#Vv1 = 1660.0 # km/s
#c1 = 4.63
#Ntot = int(128**3)
#eps = 9.3 # kpc


### eddington_nfw_far_80_in ##
#infile = "test3/eddington_nfw_far_80_in"
#Mv1 = 2.731387826049509 # from 4/3 * np.pi * rv1**3 * 200 * rhocrit
#rv1 = 63.76613181842204 # found from c1 * rs1 
#rhos1 = 1.490000e-04
#rs1 = 1.114356e+01
#Vv1 = 100.0
#Ntot = 524288
#c1 = 5.722240632115952 # found by solving rhos1 = 200/3 * rhocrit * c**3 * gc(c)
#dt = 10.0
#boxside = 10000.0
#rmax1 = 100.0 * rs1



##-- Set the IC file --##
Ntot = readGadget1.readHeader(filename=infile, strname='npart')
Mtot = readGadget1.readHeader(filename=infile, strname='mass')
time = readGadget1.readHeader(filename=infile, strname='time')

print "Ntot = ", Ntot
print "Mtot = ", Mtot
print "time = ", time, "\n"

mp = Mtot[1] # 1e10 Msun
NDM = Ntot[1]


## Max radius in units of rs ##
rOrs_max = 1.0e4 # cut-off in the radius


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
	x_com += (x[i] * 1.0 / NDM)
	y_com += (y[i] * 1.0 / NDM)
	z_com += (z[i] * 1.0 / NDM)
	vx_com += (vel_x[i] * 1.0 / NDM)
	vy_com += (vel_y[i] * 1.0 / NDM)
	vz_com += (vel_z[i] * 1.0 / NDM)

print "r_com = ", [x_com, y_com, z_com] # difference between this value and that in hernquist_test.py is between float (former) and dobule (latter)
print "v_com = ", [vx_com, vy_com, vz_com]


#for i in range(0, NDM) :
#	x[i] -= x_com
#	y[i] -= y_com
#	z[i] -= z_com
#	vel_x[i] -= vx_com
#	vel_y[i] -= vy_com
#	vel_z[i] -= vz_com


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



## compute the total energy ##
def Epot1b(r1, r2, m) :
	dist = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2 + (r1[2] - r2[2])**2)
	if dist != 0 :
		return - G * m**2  / dist # [1e10 Msun * km^2 / s^2]
	else :
		return 0

def Ekin1b(v, m) :
	vel2 = v[0]**2 + v[1]**2 + v[2]**2
	return 0.5 * m * vel2 # [1e10 Msun * km^2 / s^2]

def computeTotEnergy(num) :
	print "\nCompute total energy..."
	Epot_tot = 0
	Ekin_tot = 0

	for i in tqdm(num) :
		ep = 0
		for k in num :
			ep += Epot1b(PPosDM[i], PPosDM[k], mp)
		Epot_tot += ep
		Ekin_tot += Ekin1b(PVelDM[i], mp)

	Epot_tot /= 2 # pairwise potential is counted twice

	print "Epot_tot = ", Epot_tot
	print "Ekin_tot = ", Ekin_tot # not in the same units as Epot_tot!!!
	print "Esum_tot = ", Ekin_tot + Epot_tot
	return


####computeTotEnergy(range(NDM))







## Plot v(r) vs r (similar to what we will do for density rho(r) vs r, radial shells)
numbins = 30
mp *= 1.0e10 # Msun

r_min = np.around(a=min(pos_r), decimals=1)
r_max = np.around(a=max(pos_r), decimals=0) + 1.0
print "\nr_min = ", r_min, " kpc"
print "r_max = ", r_max, " kpc"

r_max = rOrs_max * rs1
if r_min == 0.0 :
	r_min = 0.01
else :
	r_min = 0.1 # kpc

bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # left edges


Npart_bins = [0 for j in range(0, len(bins)-1)]
vel_bins = [[] for j in range(0, len(bins)-1)]
pos_bins = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r)) :	
	for j in range(0, len(bins)-1) :
		if pos_r[i] >= bins[j] and pos_r[i] < bins[j+1] :
			Npart_bins[j] += 1
			vel_bins[j].append(vel_r[i])
			pos_bins[j].append(pos_r[i])
			break


dNpart_bins = [np.sqrt(el) for el in Npart_bins]


mid_bins = []
dmid_bins = []
vel_mean_bins = []
dvel_mean_bins = []
rho_bins = []
drho_bins = []
for j in range(0, len(bins)-1) :
	vel_sum_j = sum(vel_bins[j]) # len(vel_bins[j]) == Npart_bins[j]
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)
	if (Npart_bins[j] != 0) :
		vel_mean_bins.append(vel_sum_j/Npart_bins[j])
		dvel_mean_bins.append(vel_sum_j/(Npart_bins[j]**1.5))
		mass_j = Npart_bins[j] * mp
		dmass_j = np.sqrt(Npart_bins[j]) * mp # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins.append(mass_j / vol_j)
		drho_bins.append(dmass_j / vol_j)
	else :
		vel_mean_bins.append(0.0)
		dvel_mean_bins.append(0.0)
		rho_bins.append(0.0)
		drho_bins.append(0.0) 


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


# s = (r / rs) / c = r / rv, since rs = rv / c
bins_new = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=1e4)

rhoNFW_th = [rhos(s=el/rv1, c=c1, Mv=Mv1*1.0e10, rv=rv1) for el in bins_new]

veldispNFW_th = [veldisps(s=el/rv1, c=c1, Vv=Vv1) for el in bins_new] # G*M has right units
vescNFW = [vesc(s=el/rv1, c=c1, Vv=Vv1) for el in bins_new]


pos_bins_tot = list(chain.from_iterable(pos_bins))
vel_bins_tot = list(chain.from_iterable(vel_bins))



r_max = float(r_max / 10.0)


##-- Plot velocity distribution --##
fig0 = plt.figure(num='vel_tot', figsize=(10, 7), dpi=100) # --> check isotropy: True!
ax0 = fig0.add_subplot(111)
ax0.set_xlabel(r'$v_i$ [km/s] ', fontsize=20)
ax0.set_ylabel(r'$f(v_i)$', fontsize=20)
ax0.set_xscale('linear')
ax0.set_yscale('linear')
#ax0.set_xlim(-2000.0, 2000.0)
n1, bins1, _ = ax0.hist(vel_x, bins=100, range=None, histtype='bar', density=True, edgecolor='blue', color='blue', alpha=0.5, label=r'$v_x$')
n2, bins2, _ = ax0.hist(vel_y, bins=100, range=None, histtype='bar', density=True, edgecolor='red', color='red', alpha=0.5, label=r'$v_y$')
n3, bins3, _ = ax0.hist(vel_z, bins=100, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'$v_z$')
ax0.grid(False)
ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
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
#ax1.set_xlim(-2500.0, 2500.0)
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
ax2.plot(bins_new, vescNFW, color ='green', linestyle = '-.', lw=2.0, label=r'$v_{\rm esc}$')
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
ax3.plot(bins_new, veldispNFW_th, color ='red', linestyle = '--', lw=2.0, label=r'analytical')
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='upper right', prop={'size': 18})
ob3 = offsetbox.AnchoredText(s=r'$t = $ ' + str(np.around(a=time, decimals=1)) + r' Gyr', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
ob3_1 = offsetbox.AnchoredText(s=r'$M_v = {0:s}$'.format(as_si(Mv1*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.0f}$'.format(rv1) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3_1)
fig3.tight_layout()
fig3.show()


fig4 = plt.figure(num='rho_vs_r', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax4.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(r_min, r_max)
#ax4.set_ylim(1.0e-1, 1.0e9)
ax4.errorbar(mid_bins, rho_bins, xerr=dmid_bins, yerr=drho_bins, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'data')
ax4.plot(bins_new, rhoNFW_th, color ='red', linestyle = '--', lw=2.0, label=r'analytical')
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.9, y=2.0e3, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=r'$t = $ ' + str(np.around(a=time, decimals=1)) + r' Gyr', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=r'$M_v = {0:s}$'.format(as_si(Mv1*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.0f}$'.format(rv1) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()


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




raw_input()



