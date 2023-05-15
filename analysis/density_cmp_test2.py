import os.path
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


def rhoH(r, M, a) :
	return M / (2.0 * np.pi) * (a / r) / (r + a)**3


def veldispH(r, M, a) :
	fH = 12.0 * r * (r + a)**3 / a**4 * np.log((r + a) / r) - r / (r + a) * (25.0 + 52.0 * (r / a) + 42.0 * (r / a)**2 + 12.0 * (r / a)**3)
	return np.sqrt(G * M / (12.0 * a) * fH)


# Hernquist and simulation parameters in code units!
Xmax = 0.99

##---------# (used for test 2) #---------#
#infile_i = "test2/hernquist_test_v1"
#M = 1.0e5 # * 1e10 Msun (total mass -> galaxy cluster)
#a = 1.0e3 # kpc (scale radius)
#eps = 12.0 # kpc
##infile_02 = "test2/stability/benchmark/out/snp_002" # 1.0
##infile_05 = "test2/stability/benchmark/out/snp_005" # 3.0
##infile_07 = "test2/stability/benchmark/out/snp_007" # 5.0
##infile_09 = "test2/stability/benchmark/out/snp_009" # 7.0
##infile_13 = "test2/stability/benchmark/out/snp_013" # 10.0

#infile_02 = "test2/scattering/benchmark_noption/out/snp_002" # 1.0
#infile_05 = "test2/scattering/benchmark_noption/out/snp_005" # 3.0
#infile_07 = "test2/scattering/benchmark_noption/out/snp_007" # 5.0
#infile_09 = "test2/scattering/benchmark_noption/out/snp_009" # 7.0
#infile_13 = "test2/scattering/benchmark_noption/out/snp_013" # 10.0


#-------# (used for stability) #-------#
infile_i = "test2/hernquist_test2_v2"
M = 1.0e4 # * 1e10 Msun
a = 225.0 # kpc
eps = 4.4 # kpc

#fn = "benchmark2"
#basedir = "test2/stability/" + fn + "/out/"
#str_out = "test2/stability/data/" + fn + "_stability.dat"

fn = "benchmark2_noption"
basedir = "test2/scattering/" + fn + "/out/"
str_out = "test2/scattering/data/" + fn + "_scattering.dat"

infile_02 = basedir + "snp_002" # 1.0
infile_05 = basedir + "snp_005" # 3.0
infile_07 = basedir + "snp_007" # 5.0
infile_09 = basedir + "snp_009" # 7.0
infile_13 = basedir + "snp_013" # 10.0


numfiles = 5


##-- Set the IC file --##
Ntot = readGadget1.readHeader(filename=infile_i, strname='npart')
Mtot = readGadget1.readHeader(filename=infile_i, strname='mass')
time = readGadget1.readHeader(filename=infile_i, strname='time')

print "Ntot = ", Ntot
print "Mtot = ", Mtot
print "time = ", time, "\n"


##-- Read the particle properties --##
PPosDM_i, PVelDM_i, _, _ = readGadget1.readSnapshot(filename=infile_i, ptype='dm', strname='pos', full=True, mass=False)
PPosDM_02, PVelDM_02, _, _ = readGadget1.readSnapshot(filename=infile_02, ptype='dm', strname='pos', full=True, mass=False)
PPosDM_05, PVelDM_05, _, _ = readGadget1.readSnapshot(filename=infile_05, ptype='dm', strname='pos', full=True, mass=False)
PPosDM_07, PVelDM_07, _, _ = readGadget1.readSnapshot(filename=infile_07, ptype='dm', strname='pos', full=True, mass=False)
PPosDM_09, PVelDM_09, _, _ = readGadget1.readSnapshot(filename=infile_09, ptype='dm', strname='pos', full=True, mass=False)
PPosDM_13, PVelDM_13, _, _ = readGadget1.readSnapshot(filename=infile_13, ptype='dm', strname='pos', full=True, mass=False)

##-- Velocity distribution --##
# (Maxwell-Boltzmann? NO, maybe because f(0.5 * v**2 - G * M / (r + a)) is more complicated)
x_i, y_i, z_i = getCoord(vec=PPosDM_i)
pos_r_i = getRad(x=x_i, y=y_i, z=z_i)

x_02, y_02, z_02 = getCoord(vec=PPosDM_02)
pos_r_02 = getRad(x=x_02, y=y_02, z=z_02)
x_05, y_05, z_05 = getCoord(vec=PPosDM_05)
pos_r_05 = getRad(x=x_05, y=y_05, z=z_05)
x_07, y_07, z_07 = getCoord(vec=PPosDM_07)
pos_r_07 = getRad(x=x_07, y=y_07, z=z_07)
x_09, y_09, z_09 = getCoord(vec=PPosDM_09)
pos_r_09 = getRad(x=x_09, y=y_09, z=z_09)
x_13, y_13, z_13 = getCoord(vec=PPosDM_13)
pos_r_13 = getRad(x=x_13, y=y_13, z=z_13)

vx_i, vy_i, vz_i = getCoord(vec=PVelDM_i)
vel_tot_i = getRad(x=vx_i, y=vy_i, z=vz_i)

vx_02, vy_02, vz_02 = getCoord(vec=PVelDM_02)
vel_tot_02 = getRad(x=vx_02, y=vy_02, z=vz_02)
vx_05, vy_05, vz_05 = getCoord(vec=PVelDM_05)
vel_tot_05 = getRad(x=vx_05, y=vy_05, z=vz_05)
vx_07, vy_07, vz_07 = getCoord(vec=PVelDM_07)
vel_tot_07 = getRad(x=vx_07, y=vy_07, z=vz_07)
vx_09, vy_09, vz_09 = getCoord(vec=PVelDM_09)
vel_tot_09 = getRad(x=vx_09, y=vy_09, z=vz_09)
vx_13, vy_13, vz_13 = getCoord(vec=PVelDM_13)
vel_tot_13 = getRad(x=vx_13, y=vy_13, z=vz_13)


# velocity decomposition in spherical coordinates
phi_i = [np.arctan(y_i[i] / x_i[i]) for i in range(0, len(pos_r_i))] 
theta_i = [np.arccos(z_i[i] / pos_r_i[i]) for i in range(0, len(pos_r_i))]

phi_02 = [np.arctan(y_02[i] / x_02[i]) for i in range(0, len(pos_r_02))] 
theta_02 = [np.arccos(z_02[i] / pos_r_02[i]) for i in range(0, len(pos_r_02))]
phi_05 = [np.arctan(y_05[i] / x_05[i]) for i in range(0, len(pos_r_05))] 
theta_05 = [np.arccos(z_05[i] / pos_r_05[i]) for i in range(0, len(pos_r_05))]
phi_07 = [np.arctan(y_07[i] / x_07[i]) for i in range(0, len(pos_r_07))] 
theta_07 = [np.arccos(z_07[i] / pos_r_07[i]) for i in range(0, len(pos_r_07))]
phi_09 = [np.arctan(y_09[i] / x_09[i]) for i in range(0, len(pos_r_09))] 
theta_09 = [np.arccos(z_09[i] / pos_r_09[i]) for i in range(0, len(pos_r_09))]
phi_13 = [np.arctan(y_13[i] / x_13[i]) for i in range(0, len(pos_r_13))] 
theta_13 = [np.arccos(z_13[i] / pos_r_13[i]) for i in range(0, len(pos_r_13))]

# my method
vel_r_i = [vx_i[i]*np.sin(theta_i[i])*np.cos(phi_i[i]) + vy_i[i]*np.sin(theta_i[i])*np.sin(phi_i[i]) + vz_i[i]*np.cos(theta_i[i]) for i in range(0, len(vel_tot_i))]

vel_r_02 = [vx_02[i]*np.sin(theta_02[i])*np.cos(phi_02[i]) + vy_02[i]*np.sin(theta_02[i])*np.sin(phi_02[i]) + vz_02[i]*np.cos(theta_02[i]) for i in range(0, len(vel_tot_02))]
vel_r_05 = [vx_05[i]*np.sin(theta_05[i])*np.cos(phi_05[i]) + vy_05[i]*np.sin(theta_05[i])*np.sin(phi_05[i]) + vz_05[i]*np.cos(theta_05[i]) for i in range(0, len(vel_tot_05))]
vel_r_07 = [vx_07[i]*np.sin(theta_07[i])*np.cos(phi_07[i]) + vy_07[i]*np.sin(theta_07[i])*np.sin(phi_07[i]) + vz_07[i]*np.cos(theta_07[i]) for i in range(0, len(vel_tot_07))]
vel_r_09 = [vx_09[i]*np.sin(theta_09[i])*np.cos(phi_09[i]) + vy_09[i]*np.sin(theta_09[i])*np.sin(phi_09[i]) + vz_09[i]*np.cos(theta_09[i]) for i in range(0, len(vel_tot_09))]
vel_r_13 = [vx_13[i]*np.sin(theta_13[i])*np.cos(phi_13[i]) + vy_13[i]*np.sin(theta_13[i])*np.sin(phi_13[i]) + vz_13[i]*np.cos(theta_13[i]) for i in range(0, len(vel_tot_13))]




## Plot v(r) vs r (similar to what we will do for density rho(r) vs r, radial shells)

numbins = 30
mp = Mtot[1] * 1.0e10 # Msun
NDM = Ntot[1]

M /= Xmax

r_min = 1.0 # kpc
r_max = 100.0 * a


bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins)


Npart_bins_i = [0 for j in range(0, len(bins)-1)]
Npart_bins_02 = [0 for j in range(0, len(bins)-1)]
Npart_bins_05 = [0 for j in range(0, len(bins)-1)]
Npart_bins_07 = [0 for j in range(0, len(bins)-1)]
Npart_bins_09 = [0 for j in range(0, len(bins)-1)]
Npart_bins_13 = [0 for j in range(0, len(bins)-1)]

vel_bins_i = [[] for j in range(0, len(bins)-1)]
vel_bins_02 = [[] for j in range(0, len(bins)-1)]
vel_bins_05 = [[] for j in range(0, len(bins)-1)]
vel_bins_07 = [[] for j in range(0, len(bins)-1)]
vel_bins_09 = [[] for j in range(0, len(bins)-1)]
vel_bins_13 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_i)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_i[i] >= bins[j] and pos_r_i[i] < bins[j+1] :
			Npart_bins_i[j] += 1
			vel_bins_i[j].append(vel_r_i[i])
		if pos_r_02[i] >= bins[j] and pos_r_02[i] < bins[j+1] :
			Npart_bins_02[j] += 1
			vel_bins_02[j].append(vel_r_02[i])
		if pos_r_05[i] >= bins[j] and pos_r_05[i] < bins[j+1] :
			Npart_bins_05[j] += 1
			vel_bins_05[j].append(vel_r_05[i])
		if pos_r_07[i] >= bins[j] and pos_r_07[i] < bins[j+1] :
			Npart_bins_07[j] += 1
			vel_bins_07[j].append(vel_r_07[i])
		if pos_r_09[i] >= bins[j] and pos_r_09[i] < bins[j+1] :
			Npart_bins_09[j] += 1
			vel_bins_09[j].append(vel_r_09[i])
		if pos_r_13[i] >= bins[j] and pos_r_13[i] < bins[j+1] :
			Npart_bins_13[j] += 1
			vel_bins_13[j].append(vel_r_13[i])


dNpart_bins_i = [np.sqrt(el) for el in Npart_bins_i]
dNpart_bins_02 = [np.sqrt(el) for el in Npart_bins_02]
dNpart_bins_05 = [np.sqrt(el) for el in Npart_bins_05]
dNpart_bins_07 = [np.sqrt(el) for el in Npart_bins_07]
dNpart_bins_09 = [np.sqrt(el) for el in Npart_bins_09]
dNpart_bins_13 = [np.sqrt(el) for el in Npart_bins_13]


mid_bins = []
dmid_bins = []

rho_bins_i = []
drho_bins_i = []
rho_bins_02 = []
drho_bins_02 = []
rho_bins_05 = []
drho_bins_05 = []
rho_bins_07 = []
drho_bins_07 = []
rho_bins_09 = []
drho_bins_09 = []
rho_bins_13 = []
drho_bins_13 = []

vel_mean_bins_i = []
dvel_mean_bins_i = []
vel_mean_bins_02 = []
dvel_mean_bins_02 = []
vel_mean_bins_05 = []
dvel_mean_bins_05 = []
vel_mean_bins_07 = []
dvel_mean_bins_07 = []
vel_mean_bins_09 = []
dvel_mean_bins_09 = []
vel_mean_bins_13 = []
dvel_mean_bins_13 = []
for j in range(0, len(bins)-1) :
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)
	vel_sum_i = sum(vel_bins_i[j])
	vel_sum_02 = sum(vel_bins_02[j])
	vel_sum_05 = sum(vel_bins_05[j])
	vel_sum_07 = sum(vel_bins_07[j])
	vel_sum_09 = sum(vel_bins_09[j])
	vel_sum_13 = sum(vel_bins_13[j])
	if (Npart_bins_i[j] != 0) :
		mass_j_i = Npart_bins_i[j] * mp
		dmass_j_i = np.sqrt(Npart_bins_i[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_i.append(mass_j_i / vol_j)
		drho_bins_i.append(dmass_j_i / vol_j)
		vel_mean_bins_i.append(vel_sum_i/Npart_bins_i[j])
		dvel_mean_bins_i.append(vel_sum_i/(Npart_bins_i[j]**1.5))
	else :
		rho_bins_i.append(0.0)
		drho_bins_i.append(0.0)
		vel_mean_bins_i.append(0.0)
		dvel_mean_bins_i.append(0.0)

	if (Npart_bins_02[j] != 0) :
		mass_j_f = Npart_bins_02[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_02[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_02.append(mass_j_f / vol_j)
		drho_bins_02.append(dmass_j_f / vol_j)
		vel_mean_bins_02.append(vel_sum_02/Npart_bins_02[j])
		dvel_mean_bins_02.append(vel_sum_02/(Npart_bins_02[j]**1.5))
	else :
		rho_bins_02.append(0.0)
		drho_bins_02.append(0.0)
		vel_mean_bins_02.append(0.0)
		dvel_mean_bins_02.append(0.0) 

	if (Npart_bins_05[j] != 0) :
		mass_j_f = Npart_bins_05[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_05[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_05.append(mass_j_f / vol_j)
		drho_bins_05.append(dmass_j_f / vol_j)
		vel_mean_bins_05.append(vel_sum_05/Npart_bins_05[j])
		dvel_mean_bins_05.append(vel_sum_05/(Npart_bins_05[j]**1.5))
	else :
		rho_bins_05.append(0.0)
		drho_bins_05.append(0.0) 
		vel_mean_bins_05.append(0.0)
		dvel_mean_bins_05.append(0.0)

	if (Npart_bins_07[j] != 0) :
		mass_j_f = Npart_bins_07[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_07[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_07.append(mass_j_f / vol_j)
		drho_bins_07.append(dmass_j_f / vol_j)
		vel_mean_bins_07.append(vel_sum_07/Npart_bins_07[j])
		dvel_mean_bins_07.append(vel_sum_07/(Npart_bins_07[j]**1.5))
	else :
		rho_bins_07.append(0.0)
		drho_bins_07.append(0.0)
		vel_mean_bins_07.append(0.0)
		dvel_mean_bins_07.append(0.0) 

	if (Npart_bins_09[j] != 0) :
		mass_j_f = Npart_bins_09[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_09[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_09.append(mass_j_f / vol_j)
		drho_bins_09.append(dmass_j_f / vol_j)
		vel_mean_bins_09.append(vel_sum_09/Npart_bins_09[j])
		dvel_mean_bins_09.append(vel_sum_09/(Npart_bins_09[j]**1.5))
	else :
		rho_bins_09.append(0.0)
		drho_bins_09.append(0.0)
		vel_mean_bins_09.append(0.0)
		dvel_mean_bins_09.append(0.0) 

	if (Npart_bins_13[j] != 0) :
		mass_j_f = Npart_bins_13[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_13[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_13.append(mass_j_f / vol_j)
		drho_bins_13.append(dmass_j_f / vol_j)
		vel_mean_bins_13.append(vel_sum_13/Npart_bins_13[j])
		dvel_mean_bins_13.append(vel_sum_13/(Npart_bins_13[j]**1.5))
	else :
		rho_bins_13.append(0.0)
		drho_bins_13.append(0.0) 
		vel_mean_bins_13.append(0.0)
		dvel_mean_bins_13.append(0.0)


vel_std_bins_i = []
dvel_std_bins_i = []
for j in range(0, len(vel_bins_i)) :
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_i[j])) :
		diff2_j += (vel_bins_i[j][i] - vel_mean_bins_i[j])**2
		diff_j += (vel_bins_i[j][i] - vel_mean_bins_i[j])
	if Npart_bins_i[j] != 0 :
		vel_std_bins_i.append(np.sqrt(diff2_j / Npart_bins_i[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_i[j] * dvel_mean_bins_i[j])**2 + (diff2_j / Npart_bins_i[j]**2 * dNpart_bins_i[j])**2
			dvel_std_bins_i.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_i[j])))
		else :
			dvel_std_bins_i.append(0.0)
	else :
		vel_std_bins_i.append(0.0)
		dvel_std_bins_i.append(0.0)

vel_std_bins_02 = []
dvel_std_bins_02 = []
for j in range(0, len(vel_bins_02)) :
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_02[j])) :
		diff2_j += (vel_bins_02[j][i] - vel_mean_bins_02[j])**2
		diff_j += (vel_bins_02[j][i] - vel_mean_bins_02[j])
	if Npart_bins_02[j] != 0 :
		vel_std_bins_02.append(np.sqrt(diff2_j / Npart_bins_02[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_02[j] * dvel_mean_bins_02[j])**2 + (diff2_j / Npart_bins_02[j]**2 * dNpart_bins_02[j])**2
			dvel_std_bins_02.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_02[j])))
		else :
			dvel_std_bins_02.append(0.0)
	else :
		vel_std_bins_02.append(0.0)
		dvel_std_bins_02.append(0.0)

vel_std_bins_05 = []
dvel_std_bins_05 = []
for j in range(0, len(vel_bins_05)) :
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_05[j])) :
		diff2_j += (vel_bins_05[j][i] - vel_mean_bins_05[j])**2
		diff_j += (vel_bins_05[j][i] - vel_mean_bins_05[j])
	if Npart_bins_05[j] != 0 :
		vel_std_bins_05.append(np.sqrt(diff2_j / Npart_bins_05[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_05[j] * dvel_mean_bins_05[j])**2 + (diff2_j / Npart_bins_05[j]**2 * dNpart_bins_05[j])**2
			dvel_std_bins_05.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_05[j])))
		else :
			dvel_std_bins_05.append(0.0)
	else :
		vel_std_bins_05.append(0.0)
		dvel_std_bins_05.append(0.0)

vel_std_bins_07 = []
dvel_std_bins_07 = []
for j in range(0, len(vel_bins_07)) :
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_07[j])) :
		diff2_j += (vel_bins_07[j][i] - vel_mean_bins_07[j])**2
		diff_j += (vel_bins_07[j][i] - vel_mean_bins_07[j])
	if Npart_bins_07[j] != 0 :
		vel_std_bins_07.append(np.sqrt(diff2_j / Npart_bins_07[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_07[j] * dvel_mean_bins_07[j])**2 + (diff2_j / Npart_bins_07[j]**2 * dNpart_bins_07[j])**2
			dvel_std_bins_07.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_07[j])))
		else :
			dvel_std_bins_07.append(0.0)
	else :
		vel_std_bins_07.append(0.0)
		dvel_std_bins_07.append(0.0)

vel_std_bins_09 = []
dvel_std_bins_09 = []
for j in range(0, len(vel_bins_09)) :
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_09[j])) :
		diff2_j += (vel_bins_09[j][i] - vel_mean_bins_09[j])**2
		diff_j += (vel_bins_09[j][i] - vel_mean_bins_09[j])
	if Npart_bins_09[j] != 0 :
		vel_std_bins_09.append(np.sqrt(diff2_j / Npart_bins_09[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_09[j] * dvel_mean_bins_09[j])**2 + (diff2_j / Npart_bins_09[j]**2 * dNpart_bins_09[j])**2
			dvel_std_bins_09.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_09[j])))
		else :
			dvel_std_bins_09.append(0.0)
	else :
		vel_std_bins_09.append(0.0)
		dvel_std_bins_09.append(0.0)

vel_std_bins_13 = []
dvel_std_bins_13 = []
for j in range(0, len(vel_bins_13)) :
	diff2_j = 0
	diff_j = 0
	for i in range(0, len(vel_bins_13[j])) :
		diff2_j += (vel_bins_13[j][i] - vel_mean_bins_13[j])**2
		diff_j += (vel_bins_13[j][i] - vel_mean_bins_13[j])
	if Npart_bins_13[j] != 0 :
		vel_std_bins_13.append(np.sqrt(diff2_j / Npart_bins_13[j]))
		if diff2_j != 0 :
			tmp_j = (2.0 * diff_j / Npart_bins_13[j] * dvel_mean_bins_13[j])**2 + (diff2_j / Npart_bins_13[j]**2 * dNpart_bins_13[j])**2
			dvel_std_bins_13.append(np.sqrt(tmp_j) / (2.0 * np.sqrt(diff2_j / Npart_bins_13[j])))
		else :
			dvel_std_bins_13.append(0.0)
	else :
		vel_std_bins_13.append(0.0)
		dvel_std_bins_13.append(0.0)



## save the information for each time ##
file_out = open(str_out, "w+")
#file_out.write("mid_bins\tdmid_bins\trho_bins_i\tdrho_bins_i\trho_bins_02\tdrho_bins_02\trho_bins_05\tdrho_bins_05\trho_bins_07\tdrho_bins_07\trho_bins_09\tdrho_bins_09\trho_bins_13\tdrho_bins_13\n")
for i in range(0, len(mid_bins)) :
	file_out.write("%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(mid_bins[i], dmid_bins[i], rho_bins_i[i], drho_bins_i[i], rho_bins_02[i], drho_bins_02[i], rho_bins_05[i], drho_bins_05[i], rho_bins_07[i], drho_bins_07[i], rho_bins_09[i], drho_bins_09[i], rho_bins_13[i], drho_bins_13[i]))
file_out.close()



pt_bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=1000)


rhoH_th = [rhoH(r=pt_bins[j], M=M*1.0e10, a=a) for j in range(0, len(pt_bins))]
veldispH_th = [veldispH(r=pt_bins[j], M=M, a=a) for j in range(0, len(pt_bins))]



# for fit cored Hernquist profile
def rhoCore(r, rcore, beta, M=M*1.0e10, a=a) :
	return M / (2.0 * np.pi) * a / (r**beta + rcore**beta)**(1.0/beta) / (r + a)**3 


coeffrhoCore_02 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins, ydata=rho_bins_02, p0=[0.1*a, 4.0], sigma=drho_bins_02)
print "\nfitted [rcore, beta]_02 = ", coeffrhoCore_02[0]
fit_rho_02 = [rhoCore(el, coeffrhoCore_02[0][0], coeffrhoCore_02[0][1]) for el in pt_bins]

coeffrhoCore_05 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins, ydata=rho_bins_05, p0=[0.1*a, 4.0], sigma=drho_bins_05)
print "\nfitted [rcore, beta]_05 = ", coeffrhoCore_05[0]
fit_rho_05 = [rhoCore(el, coeffrhoCore_05[0][0], coeffrhoCore_05[0][1]) for el in pt_bins]

coeffrhoCore_07 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins, ydata=rho_bins_07, p0=[0.1*a, 4.0], sigma=drho_bins_07)
print "\nfitted [rcore, beta]_07 = ", coeffrhoCore_07[0]
fit_rho_07 = [rhoCore(el, coeffrhoCore_07[0][0], coeffrhoCore_07[0][1]) for el in pt_bins]

coeffrhoCore_09 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins, ydata=rho_bins_09, p0=[0.1*a, 4.0], sigma=drho_bins_09)
print "\nfitted [rcore, beta]_09 = ", coeffrhoCore_09[0]
fit_rho_09 = [rhoCore(el, coeffrhoCore_09[0][0], coeffrhoCore_09[0][1]) for el in pt_bins]

coeffrhoCore_13 = curve_fit(f=lambda r, rcore, beta: rhoCore(r, rcore, beta), xdata=mid_bins, ydata=rho_bins_13, p0=[0.1*a, 4.0], sigma=drho_bins_13)
print "\nfitted [rcore, beta]_13 = ", coeffrhoCore_13[0]
fit_rho_13 = [rhoCore(el, coeffrhoCore_13[0][0], coeffrhoCore_13[0][1]) for el in pt_bins]



r_max = float(r_max / 10.0)


##-- Plot velocity distribution --##
fig3 = plt.figure(num='stdv_vs_r_cmp', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax3.set_ylabel(r'$\sigma_{v_r} (r)$ [km/s]', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(r_min, r_max)
ax3.set_ylim(1.0e1, 5.0e3)
ax3.errorbar(mid_bins, vel_std_bins_i, xerr=0, yerr=dvel_std_bins_i, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax3.errorbar(mid_bins, vel_std_bins_02, xerr=0, yerr=dvel_std_bins_02, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax3.errorbar(mid_bins, vel_std_bins_05, xerr=0, yerr=dvel_std_bins_05, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax3.errorbar(mid_bins, vel_std_bins_07, xerr=0, yerr=dvel_std_bins_07, c='yellow', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax3.errorbar(mid_bins, vel_std_bins_09, xerr=0, yerr=dvel_std_bins_09, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax3.errorbar(mid_bins, vel_std_bins_13, xerr=0, yerr=dvel_std_bins_13, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax3.plot(pt_bins, veldispH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax3.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax3.text(x=eps + 0.9, y=5.0e1, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='upper right', prop={'size': 18})
ob3 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
#ob3_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob3_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
#ax3.add_artist(ob3_1)
fig3.tight_layout()
fig3.show()
#fig3.savefig('test2/figs/stdv_vs_r_cmp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig3.savefig('test2/figs/stdv_vs_r_cmp_scatt.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig3.savefig('test2/figs/stdv_vs_r_cmp2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig3.savefig('test2/figs/stdv_vs_r_cmp_scatt2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



fig4 = plt.figure(num='rho_vs_r_cmp', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax4.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(1.0e0, 2.9e2)
ax4.set_ylim(0.8e5, 4.0e8)
ax4.errorbar(mid_bins, rho_bins_i, xerr=0, yerr=drho_bins_i, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax4.errorbar(mid_bins, rho_bins_02, xerr=0, yerr=drho_bins_02, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax4.errorbar(mid_bins, rho_bins_05, xerr=0, yerr=drho_bins_05, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax4.errorbar(mid_bins, rho_bins_07, xerr=0, yerr=drho_bins_07, c='yellow', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax4.errorbar(mid_bins, rho_bins_09, xerr=0, yerr=drho_bins_09, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax4.errorbar(mid_bins, rho_bins_13, xerr=0, yerr=drho_bins_13, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax4.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.5, y=2.0e6, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
#ax4.axvline(2.0*eps, color='pink', linestyle = '-.', lw=2.0)
#ax4.text(x=2.0*eps + 0.5, y=2.0e6, s=r'$2 \epsilon$', rotation=0, color='pink', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ax4.legend(loc='upper right', prop={'size': 18})
#ob4_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
#ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()
#fig4.savefig('test2/figs/rho_vs_r_cmp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig4.savefig('test2/figs/rho_vs_r_cmp_scatt.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig4.savefig('test2/figs/rho_vs_r_cmp2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig4.savefig('test2/figs/rho_vs_r_cmp_scatt2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


fig5 = plt.figure(num='rho_vs_r_cmp_scatt', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlim(1.0e0, 2.9e2)
ax5.set_ylim(0.8e5, 4.0e8)
ax5.errorbar(mid_bins, rho_bins_i, xerr=0, yerr=drho_bins_i, c='blue', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax5.errorbar(mid_bins, rho_bins_02, xerr=0, yerr=drho_bins_02, c='cyan', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax5.errorbar(mid_bins, rho_bins_05, xerr=0, yerr=drho_bins_05, c='green', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax5.errorbar(mid_bins, rho_bins_07, xerr=0, yerr=drho_bins_07, c='yellow', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax5.errorbar(mid_bins, rho_bins_09, xerr=0, yerr=drho_bins_09, c='orange', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax5.errorbar(mid_bins, rho_bins_13, xerr=0, yerr=drho_bins_13, c='red', marker='o', ms=7.5, mec='black', mew=0.25, alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax5.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax5.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax5.text(x=eps + 0.5, y=2.0e6, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax5.plot(pt_bins, fit_rho_02, color ='cyan', linestyle = '-', lw=1.5)
ax5.plot(pt_bins, fit_rho_05, color ='green', linestyle = '-', lw=1.5)
ax5.plot(pt_bins, fit_rho_07, color ='yellow', linestyle = '-', lw=1.5)
ax5.plot(pt_bins, fit_rho_09, color ='orange', linestyle = '-', lw=1.5)
ax5.plot(pt_bins, fit_rho_13, color ='red', linestyle = '-', lw=1.5)
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
ob5_1 = offsetbox.AnchoredText(s=r'$\sigma/m_{\chi} = 1.0$ cm$^{2}/$g', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()
#fig5.savefig('test2/figs/rho_vs_r_cmp_scatt2_1.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


raw_input()
