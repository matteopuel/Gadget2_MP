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


sigmav_ann = 100.0 # cm^2/g * km/s
sigmav_ann *= UnitMass_in_g / UnitLength_in_cm**2 * 1.0e5 / UnitVelocity_in_cm_per_s # (code units)
print "sigmav_ann = ", sigmav_ann


def rhoH(roa, M, a) :
	return M / (2.0 * np.pi) / roa / (1.0 + roa)**3


#def veldispH(roa, M, a) :
#	fH = 12.0 * roa * (1.0 + roa)**3 * np.log(1.0 + 1.0 / roa) - 1.0 / (1.0 + 1.0 / roa) * (25.0 + 52.0 * roa + 42.0 * roa**2 + 12.0 * roa**3)
#	return np.sqrt(G * M / (12.0 * a) * fH)


def GammaH(roa, M, a) :
	#vpair = 4.0 / np.sqrt(np.pi) * veldispH(roa, M, a)
	#return ScatteringCrossSection * vpair * rhoH(roa, M, a)
	return sigmav_ann * rhoH(roa, M, a)



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

#fn = "benchmark2_OPTION"
#fn = "benchmark2_OPTION_eps4"
#fn = "benchmark2_OPTION_2eps"
#fn = "benchmark2_OPTION_4eps"
fn = "benchmark2_OPTION_8eps"

basedir = "test2/annihilation/" + fn + "/out/"
str_out = "test2/annihilation/data/" + fn + ".dat"

infile_00 = basedir + "snp_000"
infile_01 = basedir + "snp_001"
infile_02 = basedir + "snp_002"
infile_03 = basedir + "snp_003"
infile_04 = basedir + "snp_004"
infile_05 = basedir + "snp_005"
infile_06 = basedir + "snp_006"
infile_07 = basedir + "snp_007"
infile_08 = basedir + "snp_008"
infile_09 = basedir + "snp_009"
infile_10 = basedir + "snp_010"
infile_11 = basedir + "snp_011"
infile_12 = basedir + "snp_012"


numfiles = 13
M /= Xmax


def str2float(fld) :
	return float(fld)

def getRad(x, y, z) :
	return np.sqrt(np.square(x) + np.square(y) + np.square(z))


emptylist = []

for i in range(0, 32) :
	if i < 10 :
		tmpname = basedir + "adm_log_000" + str(i) + ".txt"
		filesize = os.path.getsize(tmpname)
		if filesize == 0 :
			emptylist.append(i)
	else :
		tmpname = basedir + "adm_log_00" + str(i) + ".txt"
		filesize = os.path.getsize(tmpname)
		if filesize == 0 :
			emptylist.append(i)


idx = -1
for i in range(0, 32) :
	if i not in emptylist :
		idx = i
		if i < 10 :
			spos1_x, spos1_y, spos1_z = np.loadtxt(fname=basedir + "adm_log_000" + str(i) + ".txt", delimiter='   ', converters={0: str2float, 1: str2float, 2: str2float}, usecols = (0, 1, 2), unpack=True)
			spos2_x, spos2_y, spos2_z = np.loadtxt(fname=basedir + "adm_log_000" + str(i) + ".txt", delimiter='   ', converters={3: str2float, 4: str2float, 5: str2float}, usecols = (3, 4, 5), unpack=True)
			break
		else :
			spos1_x, spos1_y, spos1_z = np.loadtxt(fname=basedir + "adm_log_00" + str(i) + ".txt", delimiter='   ', converters={0: str2float, 1: str2float, 2: str2float}, usecols = (0, 1, 2), unpack=True)
			spos2_x, spos2_y, spos2_z = np.loadtxt(fname=basedir + "adm_log_00" + str(i) + ".txt", delimiter='   ', converters={3: str2float, 4: str2float, 5: str2float}, usecols = (3, 4, 5), unpack=True)
			break


spos1_r = getRad(x=spos1_x, y=spos1_y, z=spos1_z)
spos2_r = getRad(x=spos2_x, y=spos2_y, z=spos2_z)

for i in range(idx+1, 32) :
	if i not in emptylist :
		if i < 10 :
			spos1_x_i, spos1_y_i, spos1_z_i = np.loadtxt(fname=basedir + "adm_log_000" + str(i) + ".txt", delimiter='   ', converters={0: str2float, 1: str2float, 2: str2float}, usecols = (0, 1, 2), unpack=True)
			spos2_x_i, spos2_y_i, spos2_z_i = np.loadtxt(fname=basedir + "adm_log_000" + str(i) + ".txt", delimiter='   ', converters={3: str2float, 4: str2float, 5: str2float}, usecols = (3, 4, 5), unpack=True)
		else : 
			spos1_x_i, spos1_y_i, spos1_z_i = np.loadtxt(fname=basedir + "adm_log_00" + str(i) + ".txt", delimiter='   ', converters={0: str2float, 1: str2float, 2: str2float}, usecols = (0, 1, 2), unpack=True)
			spos2_x_i, spos2_y_i, spos2_z_i = np.loadtxt(fname=basedir + "adm_log_00" + str(i) + ".txt", delimiter='   ', converters={3: str2float, 4: str2float, 5: str2float}, usecols = (3, 4, 5), unpack=True)
		spos1_r_i = getRad(x=spos1_x_i, y=spos1_y_i, z=spos1_z_i)
		spos2_r_i = getRad(x=spos2_x_i, y=spos2_y_i, z=spos2_z_i)
		spos1_r = np.concatenate((spos1_r, spos1_r_i), axis=None)
		spos2_r = np.concatenate((spos2_r, spos2_r_i), axis=None)

spos_r = np.concatenate((spos1_r, spos2_r), axis=None) # scattered particles (they are doubled, because two particles scatter)



##-- Set the IC file --##
Ntot = readGadget1.readHeader(filename=infile_01, strname='npart')
Mtot = readGadget1.readHeader(filename=infile_01, strname='mass')

print "Ntot = ", Ntot
print "Mtot = ", Mtot


numbins = 30
mp = Mtot[1] * 1.0e10 # Msun


r_min = 5.0e-3
r_max = 5.0

epsoa = eps / a


bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # r/a


mid_bins = []
dmid_bins = []
for j in range(0, len(bins)-1) :
	width_j = bins[j+1] - bins[j]
	mid_bins.append(bins[j] + 0.5 * width_j)
	dmid_bins.append(0.5 * width_j)


Nann_bins = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(spos_r)) :	
	for j in range(0, len(bins)-1) :
		if spos_r[i]/a >= bins[j] and spos_r[i]/a < bins[j+1] :
			Nann_bins[j] += 1
			break


print "-- done first part --"


##-- Read the particle properties --##
PPosDM_00 = readGadget1.readSnapshot(filename=infile_00, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_01 = readGadget1.readSnapshot(filename=infile_01, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_02 = readGadget1.readSnapshot(filename=infile_02, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_03 = readGadget1.readSnapshot(filename=infile_03, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_04 = readGadget1.readSnapshot(filename=infile_04, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_05 = readGadget1.readSnapshot(filename=infile_05, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_06 = readGadget1.readSnapshot(filename=infile_06, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_07 = readGadget1.readSnapshot(filename=infile_07, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_08 = readGadget1.readSnapshot(filename=infile_08, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_09 = readGadget1.readSnapshot(filename=infile_09, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_10 = readGadget1.readSnapshot(filename=infile_10, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_11 = readGadget1.readSnapshot(filename=infile_11, ptype='dm', strname='pos', full=False, mass=False)
PPosDM_12 = readGadget1.readSnapshot(filename=infile_12, ptype='dm', strname='pos', full=False, mass=False)


x_00, y_00, z_00 = getCoord(vec=PPosDM_00)
pos_r_00 = getRad(x=x_00, y=y_00, z=z_00)
x_01, y_01, z_01 = getCoord(vec=PPosDM_01)
pos_r_01 = getRad(x=x_01, y=y_01, z=z_01)
x_02, y_02, z_02 = getCoord(vec=PPosDM_02)
pos_r_02 = getRad(x=x_02, y=y_02, z=z_02)
x_03, y_03, z_03 = getCoord(vec=PPosDM_03)
pos_r_03 = getRad(x=x_03, y=y_03, z=z_03)
x_04, y_04, z_04 = getCoord(vec=PPosDM_04)
pos_r_04 = getRad(x=x_04, y=y_04, z=z_04)
x_05, y_05, z_05 = getCoord(vec=PPosDM_05)
pos_r_05 = getRad(x=x_05, y=y_05, z=z_05)
x_06, y_06, z_06 = getCoord(vec=PPosDM_06)
pos_r_06 = getRad(x=x_06, y=y_06, z=z_06)
x_07, y_07, z_07 = getCoord(vec=PPosDM_07)
pos_r_07 = getRad(x=x_07, y=y_07, z=z_07)
x_08, y_08, z_08 = getCoord(vec=PPosDM_08)
pos_r_08 = getRad(x=x_08, y=y_08, z=z_08)
x_09, y_09, z_09 = getCoord(vec=PPosDM_09)
pos_r_09 = getRad(x=x_09, y=y_09, z=z_09)
x_10, y_10, z_10 = getCoord(vec=PPosDM_10)
pos_r_10 = getRad(x=x_10, y=y_10, z=z_10)
x_11, y_11, z_11 = getCoord(vec=PPosDM_11)
pos_r_11 = getRad(x=x_11, y=y_11, z=z_11)
x_12, y_12, z_12 = getCoord(vec=PPosDM_12)
pos_r_12 = getRad(x=x_12, y=y_12, z=z_12)


# read times #
time_00 = readGadget1.readHeader(filename=infile_00, strname='time')
time_01 = readGadget1.readHeader(filename=infile_01, strname='time')
time_02 = readGadget1.readHeader(filename=infile_02, strname='time')
time_03 = readGadget1.readHeader(filename=infile_03, strname='time')
time_04 = readGadget1.readHeader(filename=infile_04, strname='time')
time_05 = readGadget1.readHeader(filename=infile_05, strname='time')
time_06 = readGadget1.readHeader(filename=infile_06, strname='time')
time_07 = readGadget1.readHeader(filename=infile_07, strname='time')
time_08 = readGadget1.readHeader(filename=infile_08, strname='time')
time_09 = readGadget1.readHeader(filename=infile_09, strname='time')
time_10 = readGadget1.readHeader(filename=infile_10, strname='time')
time_11 = readGadget1.readHeader(filename=infile_11, strname='time')
time_12 = readGadget1.readHeader(filename=infile_12, strname='time')



Npart_bins_00 = [0 for j in range(0, len(bins)-1)]
Npart_bins_01 = [0 for j in range(0, len(bins)-1)]
Npart_bins_02 = [0 for j in range(0, len(bins)-1)]
Npart_bins_03 = [0 for j in range(0, len(bins)-1)]
Npart_bins_04 = [0 for j in range(0, len(bins)-1)]
Npart_bins_05 = [0 for j in range(0, len(bins)-1)]
Npart_bins_06 = [0 for j in range(0, len(bins)-1)]
Npart_bins_07 = [0 for j in range(0, len(bins)-1)]
Npart_bins_08 = [0 for j in range(0, len(bins)-1)]
Npart_bins_09 = [0 for j in range(0, len(bins)-1)]
Npart_bins_10 = [0 for j in range(0, len(bins)-1)]
Npart_bins_11 = [0 for j in range(0, len(bins)-1)]
Npart_bins_12 = [0 for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_00)) :
	for j in range(0, len(bins)-1) :
		if pos_r_00[i]/a >= bins[j] and pos_r_00[i]/a < bins[j+1] :
			Npart_bins_00[j] += 1
		if pos_r_01[i]/a >= bins[j] and pos_r_01[i]/a < bins[j+1] :
			Npart_bins_01[j] += 1
		if pos_r_02[i]/a >= bins[j] and pos_r_02[i]/a < bins[j+1] :
			Npart_bins_02[j] += 1
		if pos_r_03[i]/a >= bins[j] and pos_r_03[i]/a < bins[j+1] :
			Npart_bins_03[j] += 1
		if pos_r_04[i]/a >= bins[j] and pos_r_04[i]/a < bins[j+1] :
			Npart_bins_04[j] += 1
		if pos_r_05[i]/a >= bins[j] and pos_r_05[i]/a < bins[j+1] :
			Npart_bins_05[j] += 1
		if pos_r_06[i]/a >= bins[j] and pos_r_06[i]/a < bins[j+1] :
			Npart_bins_06[j] += 1
		if pos_r_07[i]/a >= bins[j] and pos_r_07[i]/a < bins[j+1] :
			Npart_bins_07[j] += 1
		if pos_r_08[i]/a >= bins[j] and pos_r_08[i]/a < bins[j+1] :
			Npart_bins_08[j] += 1
		if pos_r_09[i]/a >= bins[j] and pos_r_09[i]/a < bins[j+1] :
			Npart_bins_09[j] += 1
		if pos_r_10[i]/a >= bins[j] and pos_r_10[i]/a < bins[j+1] :
			Npart_bins_10[j] += 1
		if pos_r_11[i]/a >= bins[j] and pos_r_11[i]/a < bins[j+1] :
			Npart_bins_11[j] += 1
		if pos_r_12[i]/a >= bins[j] and pos_r_12[i]/a < bins[j+1] :
			Npart_bins_12[j] += 1


# arithmetic average
#Npart_mean_bins = [(Npart_bins_00[j] + Npart_bins_01[j] + Npart_bins_02[j] + Npart_bins_03[j] + Npart_bins_04[j] + Npart_bins_05[j] + Npart_bins_06[j] + Npart_bins_07[j] + Npart_bins_08[j] + Npart_bins_09[j] + Npart_bins_10[j] + Npart_bins_11[j] + Npart_bins_12[j]) / numfiles for j in range(0, len(bins)-1)]
# time average
Npart_mean_bins = [(Npart_bins_00[j]*(time_00) + Npart_bins_01[j]*(time_01 - time_00) + Npart_bins_02[j]*(time_02 - time_01) + Npart_bins_03[j]*(time_03 - time_02) + Npart_bins_04[j]*(time_04 - time_03) + Npart_bins_05[j]*(time_05 - time_04) + Npart_bins_06[j]*(time_06 - time_05) + Npart_bins_07[j]*(time_07 - time_06) + Npart_bins_08[j]*(time_08 - time_07) + Npart_bins_09[j]*(time_09 - time_08) + Npart_bins_10[j]*(time_10 - time_09) + Npart_bins_11[j]*(time_11 - time_10) + Npart_bins_12[j]*(time_12 - time_11)) / (time_12) for j in range(0, len(bins)-1)]

ratioN_bins = [float(Nann_bins[j]) / float(Npart_mean_bins[j]) for j in range(0, len(bins)-1)]


ratioN_bins_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=ratioN_bins, k=3)
ratioN_bins_1 = ratioN_bins_interp(1.0)

Gamma_bins = [ratioN_bins[j] / ratioN_bins_1 for j in range(0, len(bins)-1)]
#Gamma_bins = [ratioN_bins[j] / (Gyr / UnitTime_in_s) for j in range(0, len(bins)-1)] # per particle, per Gyr



# compute uncertainties #
dNann_bins = [np.sqrt(el) for el in Nann_bins]

dNpart_bins_00 = [np.sqrt(el) for el in Npart_bins_00]
dNpart_bins_01 = [np.sqrt(el) for el in Npart_bins_01]
dNpart_bins_02 = [np.sqrt(el) for el in Npart_bins_02]
dNpart_bins_03 = [np.sqrt(el) for el in Npart_bins_03]
dNpart_bins_04 = [np.sqrt(el) for el in Npart_bins_04]
dNpart_bins_05 = [np.sqrt(el) for el in Npart_bins_05]
dNpart_bins_06 = [np.sqrt(el) for el in Npart_bins_06]
dNpart_bins_07 = [np.sqrt(el) for el in Npart_bins_07]
dNpart_bins_08 = [np.sqrt(el) for el in Npart_bins_08]
dNpart_bins_09 = [np.sqrt(el) for el in Npart_bins_09]
dNpart_bins_10 = [np.sqrt(el) for el in Npart_bins_10]
dNpart_bins_11 = [np.sqrt(el) for el in Npart_bins_11]
dNpart_bins_12 = [np.sqrt(el) for el in Npart_bins_12]

dNpart_mean_bins = [( (dNpart_bins_00[j]*(time_00))**2 + (dNpart_bins_01[j]*(time_01 - time_00))**2 + (dNpart_bins_02[j]*(time_02 - time_01))**2 + (dNpart_bins_03[j]*(time_03 - time_02))**2 + (dNpart_bins_04[j]*(time_04 - time_03))**2 + (dNpart_bins_05[j]*(time_05 - time_04))**2 + (dNpart_bins_06[j]*(time_06 - time_05))**2 + (dNpart_bins_07[j]*(time_07 - time_06))**2 + (dNpart_bins_08[j]*(time_08 - time_07))**2 + (dNpart_bins_09[j]*(time_09 - time_08))**2 + (dNpart_bins_10[j]*(time_10 - time_09))**2 + (dNpart_bins_11[j]*(time_11 - time_10))**2 + (dNpart_bins_12[j]*(time_12 - time_11))**2 ) / (time_12)**2 for j in range(0, len(bins)-1)]

dratioN_bins = [ratioN_bins[j] * np.sqrt( (dNann_bins[j]/Nann_bins[j])**2 + (dNpart_mean_bins[j]/Npart_mean_bins[j])**2 ) for j in range(0, len(bins)-1)]

dratioN_bins_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=dratioN_bins, k=3)
dratioN_bins_1 = dratioN_bins_interp(1.0)

dGamma_bins = [Gamma_bins[j] * np.sqrt( (dratioN_bins[j]/ratioN_bins[j])**2 + (dratioN_bins_1/ratioN_bins_1)**2 ) for j in range(0, len(bins)-1)]



## save the information for each value of h_A ##
file_out = open(str_out, "w+")
#file_out.write("mid_bins\tGamma_bins\tdmid_bins\tdGamma_bins\n")
for i in range(0, len(Gamma_bins)) :
	file_out.write("%.5f\t%.5f\t%.5f\t%.5f\n" %(mid_bins[i], Gamma_bins[i], dmid_bins[i], dGamma_bins[i]))
file_out.close()



Gamma_th = [GammaH(roa=el, M=M/Xmax, a=a) / GammaH(roa=1.0, M=M/Xmax, a=a) for el in mid_bins]
#Gamma_th = [(GammaH(roa=el, M=M/Xmax, a=a) / 2.0) / Ntot[1] for el in mid_bins] # per Gyr, per particle



##-- Plots --##
fig4 = plt.figure(num='Gamma_vs_roa', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r/a$', fontsize=20)
ax4.set_ylabel(r'$\Gamma (r) / \Gamma (a)$', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(r_min, r_max)
ax4.errorbar(mid_bins, Gamma_bins, xerr=dmid_bins, yerr=dGamma_bins, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$h_S = \epsilon$')
ax4.plot(mid_bins, Gamma_th, color ='red', linestyle = '--', lw=2.0, label=r'analytical')
ax4.axvline(epsoa, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=epsoa + 1.0e-3, y=1.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=r'$M = {0:s}$'.format(as_si(M*Xmax*1.0e10, 0)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.0f}$'.format(a) + r' kpc', loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
fig4.tight_layout()
fig4.show()
#fig4.savefig('test2/figs/Gamma_vs_roa.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig4.savefig('test2/figs/Gamma_vs_roa2.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


raw_input()
