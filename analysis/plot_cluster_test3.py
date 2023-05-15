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



plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 18})
rc('text', usetex=True)
#rc('font', size=17)
#rc('legend', fontsize=15)

rc('font', family='serif', size=18)
rc('legend', fontsize=18)
#plt.style.use('classic')


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        if orig_handle[2] != " " and orig_handle[3] != " " :
        	l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], linestyle=orig_handle[1], color=orig_handle[0])
        	l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], linestyle=orig_handle[3], color=orig_handle[2])
        	return [l1, l2]
        else :
  			l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], linestyle=orig_handle[1], color=orig_handle[0], linewidth=2.0)
  			return [l1]


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


##-- NFW --##
vbar = 200.0

def M200(r200) :
	return 4.0 * np.pi / 3.0 * r200**3 * vbar * rhocrit 

def R200(m200) :
	r3 = 3.0 * m200 / (4.0 * np.pi * vbar * rhocrit)
	return r3**(1.0/3.0)


def gc(c) :
	tmp = np.log(1.0 + c) - c / (1.0 + c)
	return 1.0 / tmp

def C2(c, Mv, rv) :
	return Mv * c**2 * gc(c) / (4.0 * np.pi * rv**3)

def rhos(s, c, Mv, rv) :
	x = c * s
	return C2(c, Mv, rv) * c / (x * (1.0 + x)**2)

def rhoNFW(r, rhoss, rs) :
	return rhoss / (r / rs) / (1.0 + r / rs)**2

def veldisps(s, c, Vv) :
	x = c * s
	arg = np.pi**2 - np.log(x) - 1.0 / x - 1.0 / (1.0 + x)**2 - 6.0 / (1.0 + x) + (1.0 + 1.0 / x**2 - 4.0 / x - 2.0 / (1.0 + x)) * np.log(1.0 + x) + 3.0 * (np.log(1.0 + x))**2 + 6.0 * spence(1.0 + x)
	tmp = 0.5 * c**2 * gc(c) * s * (1.0 + x)**2 * arg
	if tmp < 0 :
		tmp = 0
	return Vv * np.sqrt(tmp)

#def C1(c, Vv) :
#	return Vv**2 * gc(c)
#
#def Phi(s, c, Vv) :
#	# s = (r / rv), but c = rv / rs
#	x = c * s
#	if x == 0 :
#		return -C1(c, Vv) * c
#	else :
#		return -C1(c, Vv) * c * np.log(1.0 + x) / x


##-- Hernquist --##
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
##---------# (A 2537 galaxy cluster) #---------##
rv1 = 2050.0 # kpc
Mv1 = M200(rv1) # * 1e10 Msun (total mass -> galaxy cluster)
rhos1 = 1.3e-4 # * 1e10 Msun/kpc^3
rs1 = 442.0 # kpc
Vv1 = 1660.0 # km/s
c1 = 4.63

eps = 9.3 # kpc

# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
## using inferred M,a from rhos1 and rs1
infile = "test3/hernquist/hernquist_v1_S1_cluster"
M = 127766.92971544608 # * 1e10 Msun
a = 594.8897476275504 # kpc

# sigma_T (better)
infile_1 = "test3/hernquist/vector/A2537_benchmark/out/snp_013"
infile_2 = "test3/hernquist/vector/A2537_51/out/snp_013"
infile_3 = "test3/hernquist/vector/A2537_43/out/snp_013"
infile_4 = "test3/hernquist/vector/A2537_34/out/snp_013"

# sigma_V (worse, but wrong)
#infile_1 = "test3/hernquist/vector/A2537_benchmark_sgv/out/snp_013"
#infile_2 = "test3/hernquist/vector/A2537_benchmark_hcs/out/snp_013"

str_out = "test3/hernquist/data/A2537_vector_sim.dat"

# NFW (Jim - vector) #
logr_SIDM = np.loadtxt(fname='../../A2537_SIDM.dat', delimiter=' ', usecols = 0)
logrho_SIDM = np.loadtxt(fname='../../A2537_SIDM.dat', delimiter=' ', usecols = 1)

logr_68MeV = np.loadtxt(fname='../../A2537_68MeV.dat', delimiter=' ', usecols = 0)
logrho_68MeV = np.loadtxt(fname='../../A2537_68MeV.dat', delimiter=' ', usecols = 1)
logr_51MeV = np.loadtxt(fname='../../A2537_51MeV.dat', delimiter=' ', usecols = 0)
logrho_51MeV = np.loadtxt(fname='../../A2537_51MeV.dat', delimiter=' ', usecols = 1)
logr_43MeV = np.loadtxt(fname='../../A2537_43MeV.dat', delimiter=' ', usecols = 0)
logrho_43MeV = np.loadtxt(fname='../../A2537_43MeV.dat', delimiter=' ', usecols = 1)
logr_34MeV = np.loadtxt(fname='../../A2537_34MeV.dat', delimiter=' ', usecols = 0)
logrho_34MeV = np.loadtxt(fname='../../A2537_34MeV.dat', delimiter=' ', usecols = 1)


## change alpha (scalar mediator) #
#infile_1 = "test3/hernquist/scalar/A2537_benchmark/out/snp_013"
#infile_2 = "test3/hernquist/scalar/A2537_0015/out/snp_013"
#infile_3 = "test3/hernquist/scalar/A2537_002/out/snp_013"
#
#infile_4 = "test3/hernquist/scalar/A2537_002/out/snp_013"
#
#str_out = "test3/hernquist/data/A2537_scalar_sim_NEW.dat"

# NFW (Jim - scalar) #
logr_001 = np.loadtxt(fname='../../A2537s_001.dat', delimiter=' ', usecols = 0)
logrho_001 = np.loadtxt(fname='../../A2537s_001.dat', delimiter=' ', usecols = 1)
logr_0015 = np.loadtxt(fname='../../A2537s_0015.dat', delimiter=' ', usecols = 0)
logrho_0015 = np.loadtxt(fname='../../A2537s_0015.dat', delimiter=' ', usecols = 1)
logr_002 = np.loadtxt(fname='../../A2537s_002.dat', delimiter=' ', usecols = 0)
logrho_002 = np.loadtxt(fname='../../A2537s_002.dat', delimiter=' ', usecols = 1)


##-- Set the IC file --##
Ntot_1 = readGadget1.readHeader(filename=infile_1, strname='npart')
Ntot_2 = readGadget1.readHeader(filename=infile_2, strname='npart')
Ntot_3 = readGadget1.readHeader(filename=infile_3, strname='npart')
Ntot_4 = readGadget1.readHeader(filename=infile_4, strname='npart')
Mtot = readGadget1.readHeader(filename=infile_1, strname='mass')
time = readGadget1.readHeader(filename=infile_1, strname='time')

print "Ntot_1 = ", Ntot_1
print "Ntot_2 = ", Ntot_2
print "Ntot_3 = ", Ntot_3
print "Ntot_4 = ", Ntot_4
print "\nMtot = ", Mtot
print "time = ", time, "\n"

mp = Mtot[1] # 1e10 Msun


# Hernquist correction
Xmax = 0.99
M /= Xmax # the mass is a bit larger if we want that M(<r_max) = mass we want


##-- Read the particle properties --##
PPosDM_1, PVelDM_1, PIdDM_1, _ = readGadget1.readSnapshot(filename=infile_1, ptype='dm', strname='full', full=True, mass=False)
PPosDM_2, PVelDM_2, PIdDM_2, _ = readGadget1.readSnapshot(filename=infile_2, ptype='dm', strname='full', full=True, mass=False)
PPosDM_3, PVelDM_3, PIdDM_3, _ = readGadget1.readSnapshot(filename=infile_3, ptype='dm', strname='full', full=True, mass=False)
PPosDM_4, PVelDM_4, PIdDM_4, _ = readGadget1.readSnapshot(filename=infile_4, ptype='dm', strname='full', full=True, mass=False)


##-- Velocity distribution --##
# (Maxwell-Boltzmann? NO, maybe because f(0.5 * v**2 - G * M / (r + a)) is more complicated)
x_1, y_1, z_1 = getCoord(vec=PPosDM_1)
vel_x_1, vel_y_1, vel_z_1 = getCoord(vec=PVelDM_1)

pos_r_1 = getRad(x=x_1, y=y_1, z=z_1)
vel_tot_1 = getRad(x=vel_x_1, y=vel_y_1, z=vel_z_1)


x_2, y_2, z_2 = getCoord(vec=PPosDM_2)
vel_x_2, vel_y_2, vel_z_2 = getCoord(vec=PVelDM_2)

pos_r_2 = getRad(x=x_2, y=y_2, z=z_2)
vel_tot_2 = getRad(x=vel_x_2, y=vel_y_2, z=vel_z_2)


x_3, y_3, z_3 = getCoord(vec=PPosDM_3)
vel_x_3, vel_y_3, vel_z_3 = getCoord(vec=PVelDM_3)

pos_r_3 = getRad(x=x_3, y=y_3, z=z_3)
vel_tot_3 = getRad(x=vel_x_3, y=vel_y_3, z=vel_z_3)


x_4, y_4, z_4 = getCoord(vec=PPosDM_4)
vel_x_4, vel_y_4, vel_z_4 = getCoord(vec=PVelDM_4)

pos_r_4 = getRad(x=x_4, y=y_4, z=z_4)
vel_tot_4 = getRad(x=vel_x_4, y=vel_y_4, z=vel_z_4)


# velocity decomposition in spherical coordinates
phi_1 = [np.arctan(y_1[i] / x_1[i]) for i in range(0, len(pos_r_1))] 
theta_1 = [np.arccos(z_1[i] / pos_r_1[i]) for i in range(0, len(pos_r_1))]

phi_2 = [np.arctan(y_2[i] / x_2[i]) for i in range(0, len(pos_r_2))] 
theta_2 = [np.arccos(z_2[i] / pos_r_2[i]) for i in range(0, len(pos_r_2))]

phi_3 = [np.arctan(y_3[i] / x_3[i]) for i in range(0, len(pos_r_3))] 
theta_3 = [np.arccos(z_3[i] / pos_r_3[i]) for i in range(0, len(pos_r_3))]

phi_4 = [np.arctan(y_4[i] / x_4[i]) for i in range(0, len(pos_r_4))] 
theta_4 = [np.arccos(z_4[i] / pos_r_4[i]) for i in range(0, len(pos_r_4))]


# my method
vel_r_1 = [vel_x_1[i]*np.sin(theta_1[i])*np.cos(phi_1[i]) + vel_y_1[i]*np.sin(theta_1[i])*np.sin(phi_1[i]) + vel_z_1[i]*np.cos(theta_1[i]) for i in range(0, len(vel_tot_1))]
vel_theta_1 = [vel_x_1[i]*np.cos(theta_1[i])*np.cos(phi_1[i]) + vel_y_1[i]*np.cos(theta_1[i])*np.sin(phi_1[i]) - vel_z_1[i]*np.sin(theta_1[i]) for i in range(0, len(vel_tot_1))]
vel_phi_1 = [-vel_x_1[i]*np.sin(phi_1[i]) + vel_y_1[i]*np.cos(phi_1[i]) for i in range(0, len(vel_tot_1))] 

vel_r_2 = [vel_x_2[i]*np.sin(theta_2[i])*np.cos(phi_2[i]) + vel_y_2[i]*np.sin(theta_2[i])*np.sin(phi_2[i]) + vel_z_2[i]*np.cos(theta_2[i]) for i in range(0, len(vel_tot_2))]
vel_theta_2 = [vel_x_2[i]*np.cos(theta_2[i])*np.cos(phi_2[i]) + vel_y_2[i]*np.cos(theta_2[i])*np.sin(phi_2[i]) - vel_z_2[i]*np.sin(theta_2[i]) for i in range(0, len(vel_tot_2))]
vel_phi_2 = [-vel_x_2[i]*np.sin(phi_2[i]) + vel_y_2[i]*np.cos(phi_2[i]) for i in range(0, len(vel_tot_2))] 

vel_r_3 = [vel_x_3[i]*np.sin(theta_3[i])*np.cos(phi_3[i]) + vel_y_3[i]*np.sin(theta_3[i])*np.sin(phi_3[i]) + vel_z_3[i]*np.cos(theta_3[i]) for i in range(0, len(vel_tot_3))]
vel_theta_3 = [vel_x_3[i]*np.cos(theta_3[i])*np.cos(phi_3[i]) + vel_y_3[i]*np.cos(theta_3[i])*np.sin(phi_3[i]) - vel_z_3[i]*np.sin(theta_3[i]) for i in range(0, len(vel_tot_3))]
vel_phi_3 = [-vel_x_3[i]*np.sin(phi_3[i]) + vel_y_3[i]*np.cos(phi_3[i]) for i in range(0, len(vel_tot_3))] 

vel_r_4 = [vel_x_4[i]*np.sin(theta_4[i])*np.cos(phi_4[i]) + vel_y_4[i]*np.sin(theta_4[i])*np.sin(phi_4[i]) + vel_z_4[i]*np.cos(theta_4[i]) for i in range(0, len(vel_tot_4))]
vel_theta_4 = [vel_x_4[i]*np.cos(theta_4[i])*np.cos(phi_4[i]) + vel_y_4[i]*np.cos(theta_4[i])*np.sin(phi_4[i]) - vel_z_4[i]*np.sin(theta_4[i]) for i in range(0, len(vel_tot_4))]
vel_phi_4 = [-vel_x_4[i]*np.sin(phi_4[i]) + vel_y_4[i]*np.cos(phi_4[i]) for i in range(0, len(vel_tot_4))] 



## Plot v(r) vs r (similar to what we will do for density rho(r) vs r, radial shells)
numbins = 30
mp *= 1.0e10 # Msun

r_min_1 = np.around(a=min(pos_r_1), decimals=1)
r_max_1 = np.around(a=max(pos_r_1), decimals=0) + 1.0
print "\nr_min_1 = ", r_min_1, " kpc"
print "r_max_1 = ", r_max_1, " kpc"

r_min_2 = np.around(a=min(pos_r_2), decimals=1)
r_max_2 = np.around(a=max(pos_r_2), decimals=0) + 1.0
print "\nr_min_2 = ", r_min_2, " kpc"
print "r_max_2 = ", r_max_2, " kpc"

r_min_3 = np.around(a=min(pos_r_3), decimals=1)
r_max_3 = np.around(a=max(pos_r_3), decimals=0) + 1.0
print "\nr_min_3 = ", r_min_3, " kpc"
print "r_max_3 = ", r_max_3, " kpc"

r_min_4 = np.around(a=min(pos_r_4), decimals=1)
r_max_4 = np.around(a=max(pos_r_4), decimals=0) + 1.0
print "\nr_min_4 = ", r_min_4, " kpc"
print "r_max_4 = ", r_max_4, " kpc"


r_max = 100.0 * a
if r_min_1 == 0.0 or r_min_2 == 0.0 or r_min_3 == 0.0 or r_min_4 == 0.0 :
	r_min = 0.05
else :
	r_min = 1.0 # kpc


bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # left edges


#- 68 MeV -#
Npart_bins_1 = [0 for j in range(0, len(bins)-1)]
vel_bins_1 = [[] for j in range(0, len(bins)-1)]
pos_bins_1 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_1)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_1[i] >= bins[j] and pos_r_1[i] < bins[j+1] :
			Npart_bins_1[j] += 1
			vel_bins_1[j].append(vel_r_1[i])
			pos_bins_1[j].append(pos_r_1[i])
			break

dNpart_bins_1 = [np.sqrt(el) for el in Npart_bins_1]


mid_bins = []
dmid_bins = []
vel_mean_bins_1 = []
dvel_mean_bins_1 = []
rho_bins_1 = []
drho_bins_1 = []
for j in range(0, len(bins)-1) :
	vel_sum_j = sum(vel_bins_1[j]) # len(vel_bins[j]) == Npart_bins[j]
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)
	if (Npart_bins_1[j] != 0) :
		vel_mean_bins_1.append(vel_sum_j/Npart_bins_1[j])
		dvel_mean_bins_1.append(vel_sum_j/(Npart_bins_1[j]**1.5))
		mass_j = Npart_bins_1[j] * mp
		dmass_j = np.sqrt(Npart_bins_1[j]) * mp # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_1.append(mass_j / vol_j)
		drho_bins_1.append(dmass_j / vol_j)
	else :
		vel_mean_bins_1.append(0.0)
		dvel_mean_bins_1.append(0.0)
		rho_bins_1.append(0.0)
		drho_bins_1.append(0.0) 


vel_std_bins_1 = []
dvel_std_bins_1 = []
for j in range(0, len(vel_bins_1)) : # len(vel_bins) == len(bins)-1
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_1[j])) :
		diff2_j += (vel_bins_1[j][i] - vel_mean_bins_1[j])**2
		diff_j += (vel_bins_1[j][i] - vel_mean_bins_1[j])
	if Npart_bins_1[j] != 0 :
		vel_std_bins_1.append(np.sqrt(diff2_j / Npart_bins_1[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_1[j] * dvel_mean_bins_1[j])**2 + (diff2_j / Npart_bins_1[j]**2 * dNpart_bins_1[j])**2
			dvel_std_bins_1.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_1[j])))
		else :
			dvel_std_bins_1.append(0.0)
	else :
		vel_std_bins_1.append(0.0)
		dvel_std_bins_1.append(0.0)



#- 51 MeV -#
Npart_bins_2 = [0 for j in range(0, len(bins)-1)]
vel_bins_2 = [[] for j in range(0, len(bins)-1)]
pos_bins_2 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_2)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_2[i] >= bins[j] and pos_r_2[i] < bins[j+1] :
			Npart_bins_2[j] += 1
			vel_bins_2[j].append(vel_r_2[i])
			pos_bins_2[j].append(pos_r_2[i])
			break

dNpart_bins_2 = [np.sqrt(el) for el in Npart_bins_2]


vel_mean_bins_2 = []
dvel_mean_bins_2 = []
rho_bins_2 = []
drho_bins_2 = []
for j in range(0, len(bins)-1) :
	vel_sum_j = sum(vel_bins_2[j]) # len(vel_bins[j]) == Npart_bins[j]
	if (Npart_bins_2[j] != 0) :
		vel_mean_bins_2.append(vel_sum_j/Npart_bins_2[j])
		dvel_mean_bins_2.append(vel_sum_j/(Npart_bins_2[j]**1.5))
		mass_j = Npart_bins_2[j] * mp
		dmass_j = np.sqrt(Npart_bins_2[j]) * mp # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_2.append(mass_j / vol_j)
		drho_bins_2.append(dmass_j / vol_j)
	else :
		vel_mean_bins_2.append(0.0)
		dvel_mean_bins_2.append(0.0)
		rho_bins_2.append(0.0)
		drho_bins_2.append(0.0) 


vel_std_bins_2 = []
dvel_std_bins_2 = []
for j in range(0, len(vel_bins_2)) : # len(vel_bins) == len(bins)-1
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_2[j])) :
		diff2_j += (vel_bins_2[j][i] - vel_mean_bins_2[j])**2
		diff_j += (vel_bins_2[j][i] - vel_mean_bins_2[j])
	if Npart_bins_2[j] != 0 :
		vel_std_bins_2.append(np.sqrt(diff2_j / Npart_bins_2[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_2[j] * dvel_mean_bins_2[j])**2 + (diff2_j / Npart_bins_2[j]**2 * dNpart_bins_2[j])**2
			dvel_std_bins_2.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_2[j])))
		else :
			dvel_std_bins_2.append(0.0)
	else :
		vel_std_bins_2.append(0.0)
		dvel_std_bins_2.append(0.0)



#- 43 MeV -#
Npart_bins_3 = [0 for j in range(0, len(bins)-1)]
vel_bins_3 = [[] for j in range(0, len(bins)-1)]
pos_bins_3 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_3)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_3[i] >= bins[j] and pos_r_3[i] < bins[j+1] :
			Npart_bins_3[j] += 1
			vel_bins_3[j].append(vel_r_3[i])
			pos_bins_3[j].append(pos_r_3[i])
			break

dNpart_bins_3 = [np.sqrt(el) for el in Npart_bins_3]


vel_mean_bins_3 = []
dvel_mean_bins_3 = []
rho_bins_3 = []
drho_bins_3 = []
for j in range(0, len(bins)-1) :
	vel_sum_j = sum(vel_bins_3[j]) # len(vel_bins[j]) == Npart_bins[j]
	if (Npart_bins_3[j] != 0) :
		vel_mean_bins_3.append(vel_sum_j/Npart_bins_3[j])
		dvel_mean_bins_3.append(vel_sum_j/(Npart_bins_3[j]**1.5))
		mass_j = Npart_bins_3[j] * mp
		dmass_j = np.sqrt(Npart_bins_3[j]) * mp # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_3.append(mass_j / vol_j)
		drho_bins_3.append(dmass_j / vol_j)
	else :
		vel_mean_bins_3.append(0.0)
		dvel_mean_bins_3.append(0.0)
		rho_bins_3.append(0.0)
		drho_bins_3.append(0.0) 


vel_std_bins_3 = []
dvel_std_bins_3 = []
for j in range(0, len(vel_bins_3)) : # len(vel_bins) == len(bins)-1
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_3[j])) :
		diff2_j += (vel_bins_3[j][i] - vel_mean_bins_3[j])**2
		diff_j += (vel_bins_3[j][i] - vel_mean_bins_3[j])
	if Npart_bins_3[j] != 0 :
		vel_std_bins_3.append(np.sqrt(diff2_j / Npart_bins_3[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_3[j] * dvel_mean_bins_3[j])**2 + (diff2_j / Npart_bins_3[j]**2 * dNpart_bins_3[j])**2
			dvel_std_bins_3.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_3[j])))
		else :
			dvel_std_bins_3.append(0.0)
	else :
		vel_std_bins_3.append(0.0)
		dvel_std_bins_3.append(0.0)


#- 34 MeV -#
Npart_bins_4 = [0 for j in range(0, len(bins)-1)]
vel_bins_4 = [[] for j in range(0, len(bins)-1)]
pos_bins_4 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_4)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_4[i] >= bins[j] and pos_r_4[i] < bins[j+1] :
			Npart_bins_4[j] += 1
			vel_bins_4[j].append(vel_r_4[i])
			pos_bins_4[j].append(pos_r_4[i])
			break

dNpart_bins_4 = [np.sqrt(el) for el in Npart_bins_4]


vel_mean_bins_4 = []
dvel_mean_bins_4 = []
rho_bins_4 = []
drho_bins_4 = []
for j in range(0, len(bins)-1) :
	vel_sum_j = sum(vel_bins_4[j]) # len(vel_bins[j]) == Npart_bins[j]
	if (Npart_bins_4[j] != 0) :
		vel_mean_bins_4.append(vel_sum_j/Npart_bins_4[j])
		dvel_mean_bins_4.append(vel_sum_j/(Npart_bins_4[j]**1.5))
		mass_j = Npart_bins_4[j] * mp
		dmass_j = np.sqrt(Npart_bins_4[j]) * mp # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_4.append(mass_j / vol_j)
		drho_bins_4.append(dmass_j / vol_j)
	else :
		vel_mean_bins_4.append(0.0)
		dvel_mean_bins_4.append(0.0)
		rho_bins_4.append(0.0)
		drho_bins_4.append(0.0) 


vel_std_bins_4 = []
dvel_std_bins_4 = []
for j in range(0, len(vel_bins_4)) : # len(vel_bins) == len(bins)-1
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_4[j])) :
		diff2_j += (vel_bins_4[j][i] - vel_mean_bins_4[j])**2
		diff_j += (vel_bins_4[j][i] - vel_mean_bins_4[j])
	if Npart_bins_4[j] != 0 :
		vel_std_bins_4.append(np.sqrt(diff2_j / Npart_bins_4[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_4[j] * dvel_mean_bins_4[j])**2 + (diff2_j / Npart_bins_4[j]**2 * dNpart_bins_4[j])**2
			dvel_std_bins_4.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_4[j])))
		else :
			dvel_std_bins_4.append(0.0)
	else :
		vel_std_bins_4.append(0.0)
		dvel_std_bins_4.append(0.0)



# NFW
#rhoNFW_th = [rhos(s=mid_bins[j]/rv1, c=c1, Mv=Mv1*1.0e10, rv=rv1) for j in range(0, len(bins)-1)]
rhoNFW_th = [rhoNFW(r=mid_bins[j], rhoss=rhos1*1.0e10, rs=rs1) for j in range(0, len(bins)-1)]
veldispNFW_th = [veldisps(s=mid_bins[j]/(c1 * rs1), c=c1, Vv=Vv1) for j in range(0, len(bins)-1)]

# Hernquist
rhoH_th = [rhoH(r=mid_bins[j], M=M*1.0e10, a=a) for j in range(0, len(bins)-1)]
veldispH_th = [veldispH(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)] # G*M has right units #/np.sqrt(3.0) would work!
vescH = [vesc(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)]


drho_bins_1_1p = [(rho_bins_1[j] + drho_bins_1[j]) for j in range(0, len(bins)-1)]
drho_bins_1_1m = [(rho_bins_1[j] - drho_bins_1[j]) for j in range(0, len(bins)-1)]
drho_bins_2_1p = [(rho_bins_2[j] + drho_bins_2[j]) for j in range(0, len(bins)-1)]
drho_bins_2_1m = [(rho_bins_2[j] - drho_bins_2[j]) for j in range(0, len(bins)-1)]
drho_bins_3_1p = [(rho_bins_3[j] + drho_bins_3[j]) for j in range(0, len(bins)-1)]
drho_bins_3_1m = [(rho_bins_3[j] - drho_bins_3[j]) for j in range(0, len(bins)-1)]
drho_bins_4_1p = [(rho_bins_4[j] + drho_bins_4[j]) for j in range(0, len(bins)-1)]
drho_bins_4_1m = [(rho_bins_4[j] - drho_bins_4[j]) for j in range(0, len(bins)-1)]

dvel_std_bins_1_1p = [(vel_std_bins_1[j] + dvel_std_bins_1[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_1_1m = [(vel_std_bins_1[j] - dvel_std_bins_1[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_2_1p = [(vel_std_bins_2[j] + dvel_std_bins_2[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_2_1m = [(vel_std_bins_2[j] - dvel_std_bins_2[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_3_1p = [(vel_std_bins_3[j] + dvel_std_bins_3[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_3_1m = [(vel_std_bins_3[j] - dvel_std_bins_3[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_4_1p = [(vel_std_bins_4[j] + dvel_std_bins_4[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_4_1m = [(vel_std_bins_4[j] - dvel_std_bins_4[j]) for j in range(0, len(bins)-1)]


## unit conversion factors: Msun/kpc^3 -> GeV/cm^3 ##
pc = 3.08567758149e16 # m
kpc = pc * 1.0e3 / 1.0e-2 # cm  
c = 299792458.0 # m/s
eV = 1.602176634e-19 # J
GeV = eV * 1.0e9
GeVoc2 = GeV / c**2 # kg
Msun = 1.98841e30 # kg
Msun /= GeVoc2 # GeV/c^2
convfac = Msun / (kpc**3) # GeV/cm^3

print "\nMsun = ", Msun, " GeV"
print "kpc = ", kpc, " cm"

logmid_bins = [np.log10(el) for el in mid_bins]

logrho_bins_1 = [np.log10(el * convfac) if el > 0 else -2.0 for el in rho_bins_1]
logdrho_bins_1_1p = [np.log10(drho_bins_1_1p[j] * convfac) if drho_bins_1_1p[j] != 0 else 0 for j in range(0, len(bins)-1)]
logdrho_bins_1_1m = [np.log10(drho_bins_1_1m[j] * convfac) if drho_bins_1_1m[j] != 0 else 0 for j in range(0, len(bins)-1)]

logrho_bins_2 = [np.log10(el * convfac) if el > 0 else -2.0 for el in rho_bins_2]
logdrho_bins_2_1p = [np.log10(drho_bins_2_1p[j] * convfac) if drho_bins_2_1p[j] != 0 else 0 for j in range(0, len(bins)-1)]
logdrho_bins_2_1m = [np.log10(drho_bins_2_1m[j] * convfac) if drho_bins_2_1m[j] != 0 else 0 for j in range(0, len(bins)-1)]

logrho_bins_3 = [np.log10(el * convfac) if el > 0 else -2.0 for el in rho_bins_3]
logdrho_bins_3_1p = [np.log10(drho_bins_3_1p[j] * convfac) if drho_bins_3_1p[j] != 0 else 0 for j in range(0, len(bins)-1)]
logdrho_bins_3_1m = [np.log10(drho_bins_3_1m[j] * convfac) if drho_bins_3_1m[j] != 0 else 0 for j in range(0, len(bins)-1)]

logrho_bins_4 = [np.log10(el * convfac) if el > 0 else -2.0 for el in rho_bins_4]
logdrho_bins_4_1p = [np.log10(drho_bins_4_1p[j] * convfac) if drho_bins_4_1p[j] != 0 else 0 for j in range(0, len(bins)-1)]
logdrho_bins_4_1m = [np.log10(drho_bins_4_1m[j] * convfac) if drho_bins_4_1m[j] != 0 else 0 for j in range(0, len(bins)-1)]


logrhoH_th = [np.log10(el * convfac) for el in rhoH_th]
logrhoNFW_th = [np.log10(el * convfac) for el in rhoNFW_th]



## save the information in dat files ##
#file_out = open(str_out, "w+")

## VECTOR #
##file_out.write("logmid_bins\tlogrhoH_th\tlogrhoNFW_th\tlogrho_bins_1\tlogdrho_bins_1_1p\tlogdrho_bins_1_1m\tlogrho_bins_2\tlogdrho_bins_2_1p\tlogdrho_bins_2_1m\tlogrho_bins_3\tlogdrho_bins_3_1p\tlogdrho_bins_3_1m\tlogrho_bins_4\tlogdrho_bins_4_1p\tlogdrho_bins_4_1m\n")
#for i in range(0, len(logmid_bins)) :
#	file_out.write("%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(logmid_bins[i], logrhoH_th[i], logrhoNFW_th[i], logrho_bins_1[i], logdrho_bins_1_1p[i], logdrho_bins_1_1m[i], logrho_bins_2[i], logdrho_bins_2_1p[i], logdrho_bins_2_1m[i], logrho_bins_3[i], logdrho_bins_3_1p[i], logdrho_bins_3_1m[i], logrho_bins_4[i], logdrho_bins_4_1p[i], logdrho_bins_4_1m[i]))

## SCALAR #
##file_out.write("logmid_bins\tlogrhoH_th\tlogrhoNFW_th\tlogrho_bins_1\tlogdrho_bins_1_1p\tlogdrho_bins_1_1m\tlogrho_bins_2\tlogdrho_bins_2_1p\tlogdrho_bins_2_1m\tlogrho_bins_3\tlogdrho_bins_3_1p\tlogdrho_bins_3_1m\n")
#for i in range(0, len(logmid_bins)) :
#	file_out.write("%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(logmid_bins[i], logrhoH_th[i], logrhoNFW_th[i], logrho_bins_1[i], logdrho_bins_1_1p[i], logdrho_bins_1_1m[i], logrho_bins_2[i], logdrho_bins_2_1p[i], logdrho_bins_2_1m[i], logrho_bins_3[i], logdrho_bins_3_1p[i], logdrho_bins_3_1m[i]))
#
#file_out.close()



r_max = float(r_max / 10.0)


text1 = r'$r_s = {0:0.1f}$'.format(rs1) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1*1.0e10, 1)) + r' M$_{\odot}$'
text2 = r'$m_X = 100$ MeV' + '\n' + r'$\alpha\prime = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


### save vel. disp. data ##
#str_out_stdv = "test3/hernquist/data/A2537_vector_stdv.dat"
#file_out_stdv = open(str_out_stdv, "w+")
##file_out.write("mid_bins\tveldispH_th\tveldispNFW_th\tvel_std_bins_1\tdvel_std_bins_1_1p\tdvel_std_bins_1_1m\tvel_std_bins_2\tdvel_std_bins_2_1p\tdvel_std_bins_2_1m\tvel_std_bins_3\tdvel_std_bins_3_1p\tdvel_std_bins_3_1m\tvel_std_bins_4\tdvel_std_bins_4_1p\tdvel_std_bins_4_1m\n")
#for i in range(0, len(mid_bins)) :
#	file_out_stdv.write("%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(mid_bins[i], veldispH_th[i], veldispNFW_th[i], vel_std_bins_1[i], dvel_std_bins_1_1p[i], dvel_std_bins_1_1m[i], vel_std_bins_2[i], dvel_std_bins_2_1p[i], dvel_std_bins_2_1m[i], vel_std_bins_3[i], dvel_std_bins_3_1p[i], dvel_std_bins_3_1m[i], vel_std_bins_4[i], dvel_std_bins_4_1p[i], dvel_std_bins_4_1m[i]))
#file_out_stdv.close()

#str_out_stdv = "test3/hernquist/data/A2537_scalar_stdv.dat"
#file_out_stdv = open(str_out_stdv, "w+")
##file_out.write("mid_bins\tveldispH_th\tveldispNFW_th\tvel_std_bins_1\tdvel_std_bins_1_1p\tdvel_std_bins_1_1m\tvel_std_bins_2\tdvel_std_bins_2_1p\tdvel_std_bins_2_1m\tvel_std_bins_3\tdvel_std_bins_3_1p\tdvel_std_bins_3_1m\n")
#for i in range(0, len(mid_bins)) :
#	file_out_stdv.write("%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(mid_bins[i], veldispH_th[i], veldispNFW_th[i], vel_std_bins_1[i], dvel_std_bins_1_1p[i], dvel_std_bins_1_1m[i], vel_std_bins_2[i], dvel_std_bins_2_1p[i], dvel_std_bins_2_1m[i], vel_std_bins_3[i], dvel_std_bins_3_1p[i], dvel_std_bins_3_1m[i]))
#file_out_stdv.close()


##-- Plots --##
fig3 = plt.figure(num='stdv_vs_r', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax3.set_ylabel(r'$\sigma_{v_r} (r)$ [km/s]', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(2.65, r_max)
ax3.set_ylim(5.0e1, 2.0e3)
ax3.plot(mid_bins, veldispH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax3.plot(mid_bins, veldispNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax3.plot(mid_bins, vel_std_bins_1, color='blue', linestyle='-', lw=2.0, label=r'$m_V = 68$ MeV')
ax3.fill_between(mid_bins, dvel_std_bins_1_1m, dvel_std_bins_1_1p, color ='blue', alpha=0.3)
ax3.plot(mid_bins, vel_std_bins_2, color='darkviolet', linestyle='-', lw=2.0, label=r'$m_V = 51$ MeV')
ax3.fill_between(mid_bins, dvel_std_bins_2_1m, dvel_std_bins_2_1p, color ='darkviolet', alpha=0.3)
ax3.plot(mid_bins, vel_std_bins_3, color='green', linestyle='-', lw=2.0, label=r'$m_V = 43$ MeV')
ax3.fill_between(mid_bins, dvel_std_bins_3_1m, dvel_std_bins_3_1p, color ='green', alpha=0.3)
ax3.plot(mid_bins, vel_std_bins_4, color='brown', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV')
ax3.fill_between(mid_bins, dvel_std_bins_4_1m, dvel_std_bins_4_1p, color ='brown', alpha=0.3)
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='lower right', prop={'size': 16})
ob3 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
ob3_1 = offsetbox.AnchoredText(s=text1, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
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
ax4.set_xlim(2.65, r_max)
ax4.set_ylim(1.0e5, 1.0e9)
ax4.plot(mid_bins, rhoH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax4.plot(mid_bins, rhoNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax4.plot(mid_bins, rho_bins_1, color='blue', linestyle='-', lw=2.0, label=r'$m_V = 68$ MeV')
ax4.fill_between(mid_bins, drho_bins_1_1m, drho_bins_1_1p, color ='blue', alpha=0.3)
ax4.plot(mid_bins, rho_bins_2, color='darkviolet', linestyle='-', lw=2.0, label=r'$m_V = 51$ MeV')
ax4.fill_between(mid_bins, drho_bins_2_1m, drho_bins_2_1p, color ='darkviolet', alpha=0.3)
ax4.plot(mid_bins, rho_bins_3, color='green', linestyle='-', lw=2.0, label=r'$m_V = 43$ MeV')
ax4.fill_between(mid_bins, drho_bins_3_1m, drho_bins_3_1p, color ='green', alpha=0.3)
ax4.plot(mid_bins, rho_bins_4, color='brown', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV')
ax4.fill_between(mid_bins, drho_bins_4_1m, drho_bins_4_1p, color ='brown', alpha=0.3)
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.3, y=1.0e6, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 16})
ob4 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=text1, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()



fig4_1 = plt.figure(num='logrho_vs_r', figsize=(10, 7), dpi=100)
ax4_1 = fig4_1.add_subplot(111)
ax4_1.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax4_1.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax4_1.set_xscale('linear')
ax4_1.set_yscale('linear')
ax4_1.set_xlim(0.42, 2.6)
ax4_1.set_ylim(-1.9, 1.4)
ax4_1.plot(logmid_bins, logrhoH_th, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax4_1.plot(logmid_bins, logrhoNFW_th, color ='black', linestyle =':', lw=2.0, label=r'NFW')
ax4_1.plot(logr_SIDM, logrho_SIDM, color ='orange', linestyle ='-', mec='black', lw=3.0, label=r'SIDM fit 1508.03339')

ax4_1.plot(logr_68MeV, logrho_68MeV, color ='cyan', linestyle ='-.', lw=2.0, label=r'$m_V = 68$ MeV (JC, NFW)')
ax4_1.plot(logmid_bins, logrho_bins_1, color='blue', linestyle='-', lw=2.0, label=r'$m_V = 68$ MeV (MP, H)')
ax4_1.fill_between(logmid_bins, logdrho_bins_1_1m, logdrho_bins_1_1p, color ='blue', alpha=0.3)
ax4_1.plot(logr_51MeV, logrho_51MeV, color ='violet', linestyle ='-.', lw=2.0, label=r'$m_V = 51$ MeV (JC, NFW)')
ax4_1.plot(logmid_bins, logrho_bins_2, color='darkviolet', linestyle='-', lw=2.0, label=r'$m_V = 51$ MeV (MP, H)')
ax4_1.fill_between(logmid_bins, logdrho_bins_2_1m, logdrho_bins_2_1p, color ='darkviolet', alpha=0.3)
ax4_1.plot(logr_43MeV, logrho_43MeV, color ='lime', linestyle ='-.', lw=2.0, label=r'$m_V = 43$ MeV (JC, NFW)')
ax4_1.plot(logmid_bins, logrho_bins_3, color='green', linestyle='-', lw=2.0, label=r'$m_V = 43$ MeV (MP, H)')
ax4_1.fill_between(logmid_bins, logdrho_bins_3_1m, logdrho_bins_3_1p, color ='green', alpha=0.3)
ax4_1.plot(logr_34MeV, logrho_34MeV, color ='chocolate', linestyle ='-.', lw=2.0, label=r'$m_V = 34$ MeV (JC, NFW)')
ax4_1.plot(logmid_bins, logrho_bins_4, color='brown', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV (MP, H)')
ax4_1.fill_between(logmid_bins, logdrho_bins_4_1m, logdrho_bins_4_1p, color ='brown', alpha=0.3)
ax4_1.legend([("red","--", " ", " "), ("black",":", " ", " "), ("orange","-", " ", " "), ("blue","-", "cyan","-."), ("darkviolet","-", "violet","-."), ("green","-", "lime","-."), ("brown","-", "chocolate","-.")], [r'Hernquist', r'NFW', r'SIDM fit 1508.03339', r'$m_V = 68$ MeV', r'$m_V = 51$ MeV', r'$m_V = 43$ MeV', r'$m_V = 34$ MeV'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 16})

#text2 = r'$m_X = 100$ MeV' + '\n' + r'$m_V = 70$ MeV' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'
#ax4_1.set_xlim(0.58, 2.6)
#ax4_1.plot(logr_001, logrho_001, color ='cyan', linestyle ='-.', lw=2.0, label=r'$\alpha\prime = 0.01$ (JC, NFW)')
#ax4_1.plot(logmid_bins, logrho_bins_1, color='blue', linestyle='-', lw=2.0, label=r'$\alpha\prime = 0.01$ (MP, H)')
#ax4_1.fill_between(logmid_bins, logdrho_bins_1_1m, logdrho_bins_1_1p, color ='blue', alpha=0.3)
#ax4_1.plot(logr_0015, logrho_0015, color ='violet', linestyle ='-.', lw=2.0, label=r'$\alpha\prime = 0.015$ (JC, NFW)')
#ax4_1.plot(logmid_bins, logrho_bins_2, color='darkviolet', linestyle='-', lw=2.0, label=r'$\alpha\prime = 0.015$ (MP, H)')
#ax4_1.fill_between(logmid_bins, logdrho_bins_2_1m, logdrho_bins_2_1p, color ='darkviolet', alpha=0.3)
#ax4_1.plot(logr_002, logrho_002, color ='lime', linestyle ='-.', lw=2.0, label=r'$\alpha\prime = 0.02$ (JC, NFW)')
#ax4_1.plot(logmid_bins, logrho_bins_3, color='green', linestyle='-', lw=2.0, label=r'$\alpha\prime = 0.02$ (MP, H)')
#ax4_1.fill_between(logmid_bins, logdrho_bins_3_1m, logdrho_bins_3_1p, color ='green', alpha=0.3)
#ax4_1.legend([("red","--", " ", " "), ("black",":", " ", " "), ("orange","-", " ", " "), ("blue","-", "cyan","-."), ("darkviolet","-", "violet","-."), ("green","-", "lime","-.")], [r'Hernquist', r'NFW', r'SIDM fit 1508.03339', r'$\alpha\prime = 0.01$', r'$\alpha\prime = 0.015$', r'$\alpha\prime = 0.02$'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 16})
#
ax4_1.axvline(np.log10(eps), color='gray', linestyle = '-.', lw=2.0)
ax4_1.text(x=np.log10(eps) + 0.03, y=1.1, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4_1.grid(False)
ax4_1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4_1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4_1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4_1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4_1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4_1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax4_1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4_1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
#ax4_1.legend(loc='upper right', prop={'size': 16})
ob4_1 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4_1.add_artist(ob4_1)
ob4_1_1 = offsetbox.AnchoredText(s=text1, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4_1.add_artist(ob4_1_1)
fig4_1.tight_layout()
fig4_1.show()




raw_input()
