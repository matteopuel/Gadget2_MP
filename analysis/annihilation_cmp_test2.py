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


sigmav_ann = 100.0 # cm^2/g * km/s
sigmav_ann *= UnitMass_in_g / UnitLength_in_cm**2 * 1.0e5 / UnitVelocity_in_cm_per_s # (code units)


def rhoH(r, M, a) :
	return M / (2.0 * np.pi) * (a / r) / (r + a)**3

def MrH(r, M, a) :
	return M * r**3 / (r + a)**2


# Hernquist and simulation parameters in code units!
Xmax = 0.99

##---------# (used for test 2) #---------#
#infile_i = "test2/hernquist_test_v1"
#M = 1.0e5 # * 1e10 Msun (total mass -> galaxy cluster)
#a = 1.0e3 # kpc (scale radius)
#eps = 12.0 # kpc
#
##infile_02 = "test2/stability/benchmark/out/snp_002" # 1.0
##infile_05 = "test2/stability/benchmark/out/snp_005" # 3.0
##infile_07 = "test2/stability/benchmark/out/snp_007" # 5.0
##infile_09 = "test2/stability/benchmark/out/snp_009" # 7.0
##infile_13 = "test2/stability/benchmark/out/snp_013" # 10.0
#
#infile_02 = "test2/annihilation/benchmark_noption/out/snp_002" # 1.0
#infile_05 = "test2/annihilation/benchmark_noption/out/snp_005" # 3.0
#infile_07 = "test2/annihilation/benchmark_noption/out/snp_007" # 5.0
#infile_09 = "test2/annihilation/benchmark_noption/out/snp_009" # 7.0
#infile_13 = "test2/annihilation/benchmark_noption/out/snp_013" # 10.0


#-------# (used for stability) #-------#
infile_i = "test2/hernquist_test2_v2"
M = 1.0e4 # * 1e10 Msun
a = 225.0 # kpc
eps = 4.4 # kpc

#fn = "benchmark2"
#basedir = "test2/stability/" + fn + "/out/"
#str_out = "test2/annihilation/data/" + fn + "_stability.dat"

fn = "benchmark2_noption"
#fn = "benchmark2_2eps"
basedir = "test2/annihilation/" + fn + "/out/"
str_out = "test2/annihilation/data/" + fn + "_annihilation.dat"

infile_02 = basedir + "snp_002" # 1.0
infile_05 = basedir + "snp_005" # 3.0
infile_07 = basedir + "snp_007" # 5.0
infile_09 = basedir + "snp_009" # 7.0
infile_13 = basedir + "snp_013" # 10.0


##-------# (used for checks) #-------#
#infile_i = "test2/hernquist_test3_v1"
#M = 3.0 # * 1e10 Msun
#a = 10.0 # kpc
#eps = 0.3 # kpc

#infile_02 = "test2/annihilation/benchmark3_noption/out/snp_002" # 1.0
#infile_05 = "test2/annihilation/benchmark3_noption/out/snp_005" # 3.0
#infile_07 = "test2/annihilation/benchmark3_noption/out/snp_007" # 5.0
#infile_09 = "test2/annihilation/benchmark3_noption/out/snp_009" # 7.0
#infile_13 = "test2/annihilation/benchmark3_noption/out/snp_013" # 10.0


numfiles = 5


##-- Set the IC file --##
Ntot = readGadget1.readHeader(filename=infile_i, strname='npart')
Mtot = readGadget1.readHeader(filename=infile_i, strname='mass')
time = readGadget1.readHeader(filename=infile_i, strname='time')

print "Ntot = ", Ntot
print "Mtot = ", Mtot
print "time = ", time, "\n"


##-- Read the particle properties --##
PPosDM_i = readGadget1.readSnapshot(filename=infile_i, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_02 = readGadget1.readSnapshot(filename=infile_02, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_05 = readGadget1.readSnapshot(filename=infile_05, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_07 = readGadget1.readSnapshot(filename=infile_07, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_09 = readGadget1.readSnapshot(filename=infile_09, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_13 = readGadget1.readSnapshot(filename=infile_13, ptype='dm', strname='pos', full=False, mass=False)

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


# read times #
time_i = readGadget1.readHeader(filename=infile_i, strname='time')
time_02 = readGadget1.readHeader(filename=infile_02, strname='time')
time_05 = readGadget1.readHeader(filename=infile_05, strname='time')
time_07 = readGadget1.readHeader(filename=infile_07, strname='time')
time_09 = readGadget1.readHeader(filename=infile_09, strname='time')
time_13 = readGadget1.readHeader(filename=infile_13, strname='time')


#print time_i
#print time_02
#print time_05
#print time_07
#print time_09
#print time_13
#sys.exit()


## Plot v(r) vs r (similar to what we will do for density rho(r) vs r, radial shells)

numbins = 30
mp = Mtot[1] * 1.0e10 # Msun
NDM = Ntot[1]

M /= Xmax

#r_min = 1.0 # kpc
r_min = 0.1 # kpc
r_max = 100.0 * a


bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins)


Npart_bins_i = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_i)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_i[i] >= bins[j] and pos_r_i[i] < bins[j+1] :
			Npart_bins_i[j] += 1
			break

Npart_bins_02 = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_02)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_02[i] >= bins[j] and pos_r_02[i] < bins[j+1] :
			Npart_bins_02[j] += 1
			break

Npart_bins_05 = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_05)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_05[i] >= bins[j] and pos_r_05[i] < bins[j+1] :
			Npart_bins_05[j] += 1
			break

Npart_bins_07 = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_07)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_07[i] >= bins[j] and pos_r_07[i] < bins[j+1] :
			Npart_bins_07[j] += 1
			break

Npart_bins_09 = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_09)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_09[i] >= bins[j] and pos_r_09[i] < bins[j+1] :
			Npart_bins_09[j] += 1
			break

Npart_bins_13 = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_13)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_13[i] >= bins[j] and pos_r_13[i] < bins[j+1] :
			Npart_bins_13[j] += 1
			break


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

Menc_bins_i = []
dMenc_bins_i = []
menc_i = 0
dmenc2_i = 0
Menc_bins_02 = []
dMenc_bins_02 = []
menc_02 = 0
dmenc2_02 = 0
Menc_bins_05 = []
dMenc_bins_05 = []
menc_05 = 0
dmenc2_05 = 0
Menc_bins_07 = []
dMenc_bins_07 = []
menc_07 = 0
dmenc2_07 = 0
Menc_bins_09 = []
dMenc_bins_09 = []
menc_09 = 0
dmenc2_09 = 0
Menc_bins_13 = []
dMenc_bins_13 = []
menc_13 = 0
dmenc2_13 = 0

for j in range(0, len(bins)-1) :
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)
	if (Npart_bins_i[j] != 0) :
		mass_j_i = Npart_bins_i[j] * mp
		dmass_j_i = np.sqrt(Npart_bins_i[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_i.append(mass_j_i / vol_j)
		drho_bins_i.append(dmass_j_i / vol_j)
	else :
		rho_bins_i.append(0.0)
		drho_bins_i.append(0.0)

	menc_i += Npart_bins_i[j] * mp
	dmenc2_i += (np.sqrt(Npart_bins_i[j]) * mp)**2
	Menc_bins_i.append(menc_i)
	dMenc_bins_i.append(np.sqrt(dmenc2_i))

	if (Npart_bins_02[j] != 0) :
		mass_j_f = Npart_bins_02[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_02[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_02.append(mass_j_f / vol_j)
		drho_bins_02.append(dmass_j_f / vol_j)
	else :
		rho_bins_02.append(0.0)
		drho_bins_02.append(0.0)

	menc_02 += Npart_bins_02[j] * mp
	dmenc2_02 += (np.sqrt(Npart_bins_02[j]) * mp)**2
	Menc_bins_02.append(menc_02)
	dMenc_bins_02.append(np.sqrt(dmenc2_02))

	if (Npart_bins_05[j] != 0) :
		mass_j_f = Npart_bins_05[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_05[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_05.append(mass_j_f / vol_j)
		drho_bins_05.append(dmass_j_f / vol_j)
	else :
		rho_bins_05.append(0.0)
		drho_bins_05.append(0.0) 

	menc_05 += Npart_bins_05[j] * mp
	dmenc2_05 += (np.sqrt(Npart_bins_05[j]) * mp)**2
	Menc_bins_05.append(menc_05)
	dMenc_bins_05.append(np.sqrt(dmenc2_05))
		
	if (Npart_bins_07[j] != 0) :
		mass_j_f = Npart_bins_07[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_07[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_07.append(mass_j_f / vol_j)
		drho_bins_07.append(dmass_j_f / vol_j)
	else :
		rho_bins_07.append(0.0)
		drho_bins_07.append(0.0)

	menc_07 += Npart_bins_07[j] * mp
	dmenc2_07 += (np.sqrt(Npart_bins_07[j]) * mp)**2
	Menc_bins_07.append(menc_07)
	dMenc_bins_07.append(np.sqrt(dmenc2_07))
		
	if (Npart_bins_09[j] != 0) :
		mass_j_f = Npart_bins_09[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_09[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_09.append(mass_j_f / vol_j)
		drho_bins_09.append(dmass_j_f / vol_j)
	else :
		rho_bins_09.append(0.0)
		drho_bins_09.append(0.0)

	menc_09 += Npart_bins_09[j] * mp
	dmenc2_09 += (np.sqrt(Npart_bins_09[j]) * mp)**2
	Menc_bins_09.append(menc_09)
	dMenc_bins_09.append(np.sqrt(dmenc2_09))

	if (Npart_bins_13[j] != 0) :
		mass_j_f = Npart_bins_13[j] * mp
		dmass_j_f = np.sqrt(Npart_bins_13[j]) * mp # Poisson uncertainty of N counts
		vol_j= 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		rho_bins_13.append(mass_j_f / vol_j)
		drho_bins_13.append(dmass_j_f / vol_j)
	else :
		rho_bins_13.append(0.0)
		drho_bins_13.append(0.0)

	menc_13 += Npart_bins_13[j] * mp
	dmenc2_13 += (np.sqrt(Npart_bins_13[j]) * mp)**2
	Menc_bins_13.append(menc_13)
	dMenc_bins_13.append(np.sqrt(dmenc2_13)) 


Mr_bins_i = []
dMr_bins_i = []
Mr_bins_02 = []
dMr_bins_02 = []
Mr_bins_05 = []
dMr_bins_05 = []
Mr_bins_07 = []
dMr_bins_07 = []
Mr_bins_09 = []
dMr_bins_09 = []
Mr_bins_13 = []
dMr_bins_13 = []
for j in range(0, len(bins)-1) :
	mr_j_i = Menc_bins_i[j] * mid_bins[j]
	dmr_j_i = np.sqrt((mid_bins[j] * dMenc_bins_i[j])**2 + (Menc_bins_i[j] * dmid_bins[j])**2)
	Mr_bins_i.append(mr_j_i)
	dMr_bins_i.append(dmr_j_i)
	mr_j_02 = Menc_bins_02[j] * mid_bins[j]
	dmr_j_02 = np.sqrt((mid_bins[j] * dMenc_bins_02[j])**2 + (Menc_bins_02[j] * dmid_bins[j])**2)
	Mr_bins_02.append(mr_j_02)
	dMr_bins_02.append(dmr_j_02)
	mr_j_05 = Menc_bins_05[j] * mid_bins[j]
	dmr_j_05 = np.sqrt((mid_bins[j] * dMenc_bins_05[j])**2 + (Menc_bins_05[j] * dmid_bins[j])**2)
	Mr_bins_05.append(mr_j_05)
	dMr_bins_05.append(dmr_j_05)
	mr_j_07 = Menc_bins_07[j] * mid_bins[j]
	dmr_j_07 = np.sqrt((mid_bins[j] * dMenc_bins_07[j])**2 + (Menc_bins_07[j] * dmid_bins[j])**2)
	Mr_bins_07.append(mr_j_07)
	dMr_bins_07.append(dmr_j_07)
	mr_j_09 = Menc_bins_09[j] * mid_bins[j]
	dmr_j_09 = np.sqrt((mid_bins[j] * dMenc_bins_09[j])**2 + (Menc_bins_09[j] * dmid_bins[j])**2)
	Mr_bins_09.append(mr_j_09)
	dMr_bins_09.append(dmr_j_09)
	mr_j_13 = Menc_bins_13[j] * mid_bins[j]
	dmr_j_13 = np.sqrt((mid_bins[j] * dMenc_bins_13[j])**2 + (Menc_bins_13[j] * dmid_bins[j])**2)
	Mr_bins_13.append(mr_j_13)
	dMr_bins_13.append(dmr_j_13)



## save the information for each time ##
file_out = open(str_out, "w+")
#file_out.write("mid_bins\tdmid_bins\trho_bins_i\tdrho_bins_i\trho_bins_02\tdrho_bins_02\trho_bins_05\tdrho_bins_05\trho_bins_07\tdrho_bins_07\trho_bins_09\tdrho_bins_09\trho_bins_13\tdrho_bins_13\ttime_list\n")
for i in range(0, len(mid_bins)) :
	file_out.write("%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" %(mid_bins[i], dmid_bins[i], rho_bins_i[i], drho_bins_i[i], rho_bins_02[i], drho_bins_02[i], rho_bins_05[i], drho_bins_05[i], rho_bins_07[i], drho_bins_07[i], rho_bins_09[i], drho_bins_09[i], rho_bins_13[i], drho_bins_13[i]))
file_out.close()



pt_bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=1000)
rhoH_th = [rhoH(r=pt_bins[j], M=M*1.0e10, a=a) for j in range(0, len(pt_bins))]
MrH_th = [MrH(r=el, M=M*1.0e10, a=a) for el in pt_bins]



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
	if rho_bins_02[j] >= rho_bins_02[-1] : # since rho(r) is monotonically decreasing function
		mid_bins_sub_02.append(mid_bins[j])
		rho_bins_sub_02.append(rho_bins_02[j])
		drho_bins_sub_02.append(drho_bins_02[j])

	if rho_bins_05[j] >= rho_bins_05[-1] :
		mid_bins_sub_05.append(mid_bins[j])
		rho_bins_sub_05.append(rho_bins_05[j])
		drho_bins_sub_05.append(drho_bins_05[j])

	if rho_bins_07[j] >= rho_bins_07[-1] :
		mid_bins_sub_07.append(mid_bins[j])
		rho_bins_sub_07.append(rho_bins_07[j])
		drho_bins_sub_07.append(drho_bins_07[j])

	if rho_bins_09[j] >= rho_bins_09[-1] :
		mid_bins_sub_09.append(mid_bins[j])
		rho_bins_sub_09.append(rho_bins_09[j])
		drho_bins_sub_09.append(drho_bins_09[j])

	if rho_bins_13[j] >= rho_bins_13[-1] :
		mid_bins_sub_13.append(mid_bins[j])
		rho_bins_sub_13.append(rho_bins_13[j])
		drho_bins_sub_13.append(drho_bins_13[j])

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



r_max = float(r_max / 10.0)


##-- Plots --##
fig4 = plt.figure(num='rho_vs_r_cmp_ann', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax4.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(1.71e0, 6.0e2) #(1.0e0, 2.9e2)
ax4.set_ylim(0.1e5, 3.0e8)
ax4.errorbar(mid_bins, rho_bins_i, xerr=0, yerr=drho_bins_i, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax4.errorbar(mid_bins, rho_bins_02, xerr=0, yerr=drho_bins_02, c='cyan', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax4.errorbar(mid_bins, rho_bins_05, xerr=0, yerr=drho_bins_05, c='green', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax4.errorbar(mid_bins, rho_bins_07, xerr=0, yerr=drho_bins_07, c='yellow', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax4.errorbar(mid_bins, rho_bins_09, xerr=0, yerr=drho_bins_09, c='orange', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax4.errorbar(mid_bins, rho_bins_13, xerr=0, yerr=drho_bins_13, c='red', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax4.plot(pt_bins, rhoH_th, color ='black', linestyle = '--', lw=2.0)#, label=r'analytical')
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.25*np.sqrt(eps), y=3.0e5, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.plot(pt_bins, fit_rhoH_ann_02, color ='cyan', linestyle = '-', lw=1.5)
ax4.plot(pt_bins, fit_rhoH_ann_05, color ='green', linestyle = '-', lw=1.5)
ax4.plot(pt_bins, fit_rhoH_ann_07, color ='yellow', linestyle = '-', lw=1.5)
ax4.plot(pt_bins, fit_rhoH_ann_09, color ='orange', linestyle = '-', lw=1.5)
ax4.plot(pt_bins, fit_rhoH_ann_13, color ='red', linestyle = '-', lw=1.5)

ax4.plot(pt_bins, fit_rho_02, color ='cyan', linestyle = ':', lw=2.0)
ax4.plot(pt_bins, fit_rho_05, color ='green', linestyle = ':', lw=2.0)
ax4.plot(pt_bins, fit_rho_07, color ='yellow', linestyle = ':', lw=2.0)
ax4.plot(pt_bins, fit_rho_09, color ='orange', linestyle = ':', lw=2.0)
ax4.plot(pt_bins, fit_rho_13, color ='red', linestyle = ':', lw=2.0)

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
ob4_1 = offsetbox.AnchoredText(s=r'$\langle \sigma_{\rm ann} v \rangle /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()
#fig4.savefig('test2/figs/rho_vs_r_cmp_ann.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig4.savefig('test2/figs/rho_vs_r_cmp_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



fig5 = plt.figure(num='Mr_vs_r_cmp_ann', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$M(r) \cdot r$ [M$_{\odot}$ kpc]', fontsize=20)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlim(r_min, r_max)
ax5.set_ylim(5.0e7, 1.0e21)
ax5.errorbar(mid_bins, Mr_bins_i, xerr=0, yerr=dMr_bins_i, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 0$ Gyr')
ax5.errorbar(mid_bins, Mr_bins_02, xerr=0, yerr=dMr_bins_02, c='cyan', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 1$ Gyr')
ax5.errorbar(mid_bins, Mr_bins_05, xerr=0, yerr=dMr_bins_05, c='green', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 3$ Gyr')
ax5.errorbar(mid_bins, Mr_bins_07, xerr=0, yerr=dMr_bins_07, c='yellow', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 5$ Gyr')
ax5.errorbar(mid_bins, Mr_bins_09, xerr=0, yerr=dMr_bins_09, c='orange', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 7$ Gyr')
ax5.errorbar(mid_bins, Mr_bins_13, xerr=0, yerr=dMr_bins_13, c='red', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$t = 10$ Gyr')
ax5.plot(pt_bins, MrH_th, color ='black', linestyle = '--', lw=2.0, label=r'analytical')
ax5.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax5.text(x=eps + 0.25*np.sqrt(eps), y=1.0e17, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='all', numticks=15))
ax5.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax5.legend(loc='lower right', prop={'size': 18})
#ob5 = offsetbox.AnchoredText(s=r'CDM', loc='upper right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5 = offsetbox.AnchoredText(s=r'$\langle \sigma_{\rm ann} v \rangle /m_{\chi} = 100$ cm$^{2}/$g km/s', loc='upper right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
ob5_1 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='upper left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()
#fig5.savefig('test2/figs/Mr_vs_r_cmp_CDM.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig5.savefig('test2/figs/Mr_vs_r_cmp_CDM2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig5.savefig('test2/figs/Mr_vs_r_cmp_ann.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig5.savefig('test2/figs/Mr_vs_r_cmp_ann2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


raw_input()
