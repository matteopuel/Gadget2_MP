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
from scipy import interpolate, integrate
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

plt.rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amssymb}'])


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


def M200s(rhoss, rs, c) :
	return 4.0 * np.pi * rhoss * rs**3 / gc(c)

def MrNFW(r, rhoss, rs, c) :
	m200 = M200s(rhoss, rs, c)
	return m200 * gc(c) * (np.log(1.0 + r / rs) - (r / rs) / (1 + r / rs))

def vcircNFW(r, rhoss, rs, c) :
	mr = MrNFW(r, rhoss, rs, c)
	return np.sqrt(G * mr / r)


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


def vcircH(r, M, a) :
	return np.sqrt(G * M * r) / (r + a)


# Hernquist and simulation parameters in code units!
#------# (real simulation: DDO 154) #-------#
Mv1 = 2.3 # * 1e10 Msun (total mass -> dSph galaxy)
rv1 = R200(Mv1) # kpc
rhos1 = 1.5e-3 # * 1e10 Msun/kpc^3
rs1 = 3.4 # kpc
Vv1 = 49.0 # km/s
c1 = 12.2

eps = 0.3 # kpc


# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
#infile = "test3/hernquist/hernquist_v1_S"
#M = Mv1 # * 1e10 Msun
#a = 8.982285853787648 # kpc
## using inferred M,a from rhos1 and rs1
infile = "test3/hernquist/hernquist_v1_S1"
M = 1.226850076349031
a = 6.187578545092555

r_SIDM_tot = np.loadtxt(fname="../../DDO154_sidm_tot.dat", delimiter=' ', usecols = 0)
vcirc_SIDM_tot = np.loadtxt(fname="../../DDO154_sidm_tot.dat", delimiter=' ', usecols = 1)
r_SIDM_dm = np.loadtxt(fname="../../DDO154_sidm_dm.dat", delimiter=' ', usecols = 0)
vcirc_SIDM_dm = np.loadtxt(fname="../../DDO154_sidm_dm.dat", delimiter=' ', usecols = 1)


# sigma_T (worse)
infile_1 = "test3/hernquist/vector/DDO154_benchmark/out/snp_013"
infile_2 = "test3/hernquist/vector/DDO154_34/out/snp_013"
infile_3 = "test3/hernquist/vector/DDO154_17/out/snp_013"
#infile_3 = "test3/hernquist/vector/DDO154_36/out/snp_013"

## sigma_V (better, but WRONG because not normalized)
##infile_1 = "test3/hernquist/vector/DDO154_benchmark_sgv/out/snp_013"

str_out = "test3/hernquist/data/DDO154_vector_sim.dat"


## NFW (Jim - vector) #
logr_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 0)
logrho_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 1)

logr_26MeV = np.loadtxt(fname='../../DDO154_26MeV.dat', delimiter=' ', usecols = 0)
logrho_26MeV = np.loadtxt(fname='../../DDO154_26MeV.dat', delimiter=' ', usecols = 1)
logr_34MeV = np.loadtxt(fname='../../DDO154_34MeV.dat', delimiter=' ', usecols = 0)
logrho_34MeV = np.loadtxt(fname='../../DDO154_34MeV.dat', delimiter=' ', usecols = 1)
logr_17MeV = np.loadtxt(fname='../../DDO154_17MeV.dat', delimiter=' ', usecols = 0)
logrho_17MeV = np.loadtxt(fname='../../DDO154_17MeV.dat', delimiter=' ', usecols = 1)
# Hernquist (Jim - vector) #
logr_26MeV_H = np.loadtxt(fname='../../rho-dwarf-vector-hern.dat', delimiter='\t\t\t', usecols = 0)
logrho_26MeV_H = np.loadtxt(fname='../../rho-dwarf-vector-hern.dat', delimiter='\t\t\t', usecols = 2)


## change alpha (scalar mediator) #
#infile_1 = "test3/hernquist/scalar/DDO154_benchmark/out/snp_013"
#infile_2 = "test3/hernquist/scalar/DDO154_0015/out/snp_013"
#infile_3 = "test3/hernquist/scalar/DDO154_002/out/snp_013"
#
#str_out = "test3/hernquist/data/DDO154_scalar_sim_NEW.dat"


# NFW (Jim - scalar) #
logr_001 = np.loadtxt(fname='../../DDO154s_001.dat', delimiter=' ', usecols = 0)
logrho_001 = np.loadtxt(fname='../../DDO154s_001.dat', delimiter=' ', usecols = 1)
logr_0015 = np.loadtxt(fname='../../DDO154s_0015.dat', delimiter=' ', usecols = 0)
logrho_0015 = np.loadtxt(fname='../../DDO154s_0015.dat', delimiter=' ', usecols = 1)
logr_002 = np.loadtxt(fname='../../DDO154s_002.dat', delimiter=' ', usecols = 0)
logrho_002 = np.loadtxt(fname='../../DDO154s_002.dat', delimiter=' ', usecols = 1)



## REAL DATA ##
r_exp_tmp = np.loadtxt(fname='../../DDO154_data_exp_tot.dat', delimiter=' ', usecols = 0)
vc_exp_tmp = np.loadtxt(fname='../../DDO154_data_exp_tot.dat', delimiter=' ', usecols = 1)
r_exp_dm = np.loadtxt(fname='../../DDO154_data_exp_dm.dat', delimiter=' ', usecols = 0)
vc_exp_dm = np.loadtxt(fname='../../DDO154_data_exp_dm.dat', delimiter=' ', usecols = 1)
# three by three (hp: symmetric uncertainty)
r_exp_tot = []
vc_exp_tot = []
dvc_exp_tot = []
for i in range(0, len(r_exp_tmp), 3) :
	r_exp_i = (r_exp_tmp[i+0] + r_exp_tmp[i+1] + r_exp_tmp[i+2]) / 3.0
	vc_exp_i = (vc_exp_tmp[i+0] + vc_exp_tmp[i+1] + vc_exp_tmp[i+2]) / 3.0
	dvc_exp_i = (vc_exp_tmp[i+2] - vc_exp_tmp[i+0]) / 2.0
	r_exp_tot.append(r_exp_i)
	vc_exp_tot.append(vc_exp_i)
	dvc_exp_tot.append(dvc_exp_i)



### save the information in dat files ##
#str_out = "DDO154_data_tot.dat"
##str_out = "DDO126_data_tot.dat"
#file_out = open(str_out, "w+")
##file_out.write("logmid_bins\tlogrhoH_th\tlogrhoNFW_th\tlogrho_bins_1\tlogdrho_bins_1_1p\tlogdrho_bins_1_1m\tlogrho_bins_2\tlogdrho_bins_2_1p\tlogdrho_bins_2_1m\tlogrho_bins_3\tlogdrho_bins_3_1p\tlogdrho_bins_3_1m\n")
#for i in range(0, len(r_exp_tot)) :
#	file_out.write("%.15f\t%.15f\t%.15f\n" %(r_exp_tot[i], vc_exp_tot[i], dvc_exp_tot[i]))
#file_out.close()
#sys.exit()



## CONVERSION JIM'S POINTS IN VCIRC ##
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


def rho2vcirc(logr_list, logrho_list) :
	r_list = [10.0**el for el in logr_list] # kpc
	rho_list = [(10.0**el) / convfac / (1.0e10) for el in logrho_list] # 1e10 Msun/kpc^3
	Mr_list_int = [4.0 * np.pi * r_list[j]**2 * rho_list[j] for j in range(0, len(r_list))] # 1e10 Msun/kpc
	Mr_list_int_interp = interpolate.InterpolatedUnivariateSpline(x=r_list, y=Mr_list_int, k=3)
	Mr_list = [Mr_list_int_interp.integral(0, el) for el in r_list] # 1e10 Msun/kpc
	vcirc_list = [np.sqrt(G * Mr_list[j] / r_list[j]) for j in range(0, len(r_list))]
	return [r_list, vcirc_list]


# SIDM fit curve #
##r_SIDM, vcirc_SIDM = rho2vcirc(logr_list=logr_SIDM, logrho_list=logrho_SIDM)
r_SIDM = r_SIDM_dm # better than the one above
vcirc_SIDM = vcirc_SIDM_dm

# 26 MeV curve #
r_26MeV, vcirc_26MeV = rho2vcirc(logr_list=logr_26MeV, logrho_list=logrho_26MeV)
# 34 MeV curve #
r_34MeV, vcirc_34MeV = rho2vcirc(logr_list=logr_34MeV, logrho_list=logrho_34MeV)
# 17 MeV curve #
r_17MeV, vcirc_17MeV = rho2vcirc(logr_list=logr_17MeV, logrho_list=logrho_17MeV)

# 001 curve #
r_001, vcirc_001 = rho2vcirc(logr_list=logr_001, logrho_list=logrho_001)
# 0015 curve #
r_0015, vcirc_0015 = rho2vcirc(logr_list=logr_0015, logrho_list=logrho_0015)
# 002 curve #
r_002, vcirc_002 = rho2vcirc(logr_list=logr_002, logrho_list=logrho_002)



##-- Set the IC file --##
Ntot_1 = readGadget1.readHeader(filename=infile_1, strname='npart')
Ntot_2 = readGadget1.readHeader(filename=infile_2, strname='npart')
Ntot_3 = readGadget1.readHeader(filename=infile_3, strname='npart')
Mtot = readGadget1.readHeader(filename=infile_1, strname='mass')
time = readGadget1.readHeader(filename=infile_1, strname='time')

print "Ntot_26MeV = ", Ntot_1
print "Ntot_34MeV = ", Ntot_2
print "Ntot_17MeV = ", Ntot_3
print "\nMtot = ", Mtot
print "time = ", time, "\n"

mp = Mtot[1] # 1e10 Msun


# Hernquist correction
Xmax = 0.99
M /= Xmax # the mass is a bit larger if we want that M(<r_max) = mass we want


##-- Read the particle properties --##
PPosDM_1, _, _, PMassDM_1 = readGadget1.readSnapshot(filename=infile_1, ptype='dm', strname='full', full=True, mass=False)
PPosDM_2, _, _, PMassDM_2 = readGadget1.readSnapshot(filename=infile_2, ptype='dm', strname='full', full=True, mass=False)
PPosDM_3, _, _, PMassDM_3 = readGadget1.readSnapshot(filename=infile_3, ptype='dm', strname='full', full=True, mass=False)


##-- Velocity distribution --##
# (Maxwell-Boltzmann? NO, maybe because f(0.5 * v**2 - G * M / (r + a)) is more complicated)
x_1, y_1, z_1 = getCoord(vec=PPosDM_1)
pos_r_1 = getRad(x=x_1, y=y_1, z=z_1)


x_2, y_2, z_2 = getCoord(vec=PPosDM_2)
pos_r_2 = getRad(x=x_2, y=y_2, z=z_2)


x_3, y_3, z_3 = getCoord(vec=PPosDM_3)
pos_r_3 = getRad(x=x_3, y=y_3, z=z_3)



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


r_max = 100.0 * a
if r_min_1 == 0.0 or r_min_2 == 0.0 or r_min_3 == 0.0 :
	r_min = 0.05
else :
	r_min = 1.0 # kpc


#bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # left edges
r_max = 10.0
bins = np.linspace(start=r_min, stop=r_max, num=numbins) # left edges


#- 26 MeV -#
Npart_bins_1 = [0 for j in range(0, len(bins)-1)]
Mp_bins_1 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_1)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_1[i] >= bins[j] and pos_r_1[i] < bins[j+1] :
			Npart_bins_1[j] += 1
			Mp_bins_1[j].append(PMassDM_1[i])
			break


mid_bins = []
dmid_bins = []
Menc_bins_1 = []
dMenc_bins_1 = []
menc_1 = 0
dmenc2_1 = 0
for j in range(0, len(bins)-1) :
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)
	if (Npart_bins_1[j] != 0) :
		mass_j = sum(Mp_bins_1[j])
		dmass_j = np.sqrt(Npart_bins_1[j]) * (mass_j / Npart_bins_1[j]) # Poisson uncertainty of N counts
		menc_1 += mass_j
		dmenc2_1 += dmass_j**2
	else :
		menc_1 += 0.0
		dmenc2_1 += 0.0

	Menc_bins_1.append(menc_1)
	dMenc_bins_1.append(np.sqrt(dmenc2_1))


vcirc_bins_1 = []
dvcirc_bins_1 = []
for j in range(0, len(bins)-1) :
	vc_j_1 = np.sqrt(G * Menc_bins_1[j] / mid_bins[j])
	dvc_j_1 = vc_j_1 / 2.0 * np.sqrt((dMenc_bins_1[j] / Menc_bins_1[j])**2 + (dmid_bins[j] / mid_bins[j])**2)
	vcirc_bins_1.append(vc_j_1)
	dvcirc_bins_1.append(dvc_j_1)



#- 34 MeV -#
Npart_bins_2 = [0 for j in range(0, len(bins)-1)]
Mp_bins_2 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_2)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_2[i] >= bins[j] and pos_r_2[i] < bins[j+1] :
			Npart_bins_2[j] += 1
			Mp_bins_2[j].append(PMassDM_2[i])
			break


Menc_bins_2 = []
dMenc_bins_2 = []
menc_2 = 0
dmenc2_2 = 0
for j in range(0, len(bins)-1) :
	if (Npart_bins_2[j] != 0) :
		mass_j = sum(Mp_bins_2[j])
		dmass_j = np.sqrt(Npart_bins_2[j]) * (mass_j / Npart_bins_2[j]) # Poisson uncertainty of N counts
		menc_2 += mass_j
		dmenc2_2 += dmass_j**2
	else :
		menc_2 += 0.0
		dmenc2_2 += 0.0

	Menc_bins_2.append(menc_2)
	dMenc_bins_2.append(np.sqrt(dmenc2_2))


vcirc_bins_2 = []
dvcirc_bins_2 = []
for j in range(0, len(bins)-1) :
	vc_j_2 = np.sqrt(G * Menc_bins_2[j] / mid_bins[j])
	dvc_j_2 = vc_j_2 / 2.0 * np.sqrt((dMenc_bins_2[j] / Menc_bins_2[j])**2 + (dmid_bins[j] / mid_bins[j])**2)
	vcirc_bins_2.append(vc_j_2)
	dvcirc_bins_2.append(dvc_j_2)



#- 17 MeV -#
Npart_bins_3 = [0 for j in range(0, len(bins)-1)]
Mp_bins_3 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_3)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_3[i] >= bins[j] and pos_r_3[i] < bins[j+1] :
			Npart_bins_3[j] += 1
			Mp_bins_3[j].append(PMassDM_3[i])
			break


Menc_bins_3 = []
dMenc_bins_3 = []
menc_3 = 0
dmenc2_3 = 0
for j in range(0, len(bins)-1) :
	if (Npart_bins_3[j] != 0) :
		mass_j = sum(Mp_bins_3[j])
		dmass_j = np.sqrt(Npart_bins_3[j]) * (mass_j / Npart_bins_3[j]) # Poisson uncertainty of N counts
		menc_3 += mass_j
		dmenc2_3 += dmass_j**2
	else :
		menc_3 += 0.0
		dmenc2_3 += 0.0

	Menc_bins_3.append(menc_3)
	dMenc_bins_3.append(np.sqrt(dmenc2_3))


vcirc_bins_3 = []
dvcirc_bins_3 = []
for j in range(0, len(bins)-1) :
	vc_j_3 = np.sqrt(G * Menc_bins_3[j] / mid_bins[j])
	dvc_j_3 = vc_j_3 / 2.0 * np.sqrt((dMenc_bins_3[j] / Menc_bins_3[j])**2 + (dmid_bins[j] / mid_bins[j])**2)
	vcirc_bins_3.append(vc_j_3)
	dvcirc_bins_3.append(dvc_j_3)



# NFW
vcircNFW_th = [vcircNFW(r=mid_bins[j], rhoss=rhos1, rs=rs1, c=c1) for j in range(0, len(bins)-1)]

# Hernquist
vcircH_th = [vcircH(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)]


dvcirc_bins_1_1p = [(vcirc_bins_1[j] + dvcirc_bins_1[j]) for j in range(0, len(bins)-1)]
dvcirc_bins_1_1m = [(vcirc_bins_1[j] - dvcirc_bins_1[j]) for j in range(0, len(bins)-1)]
dvcirc_bins_2_1p = [(vcirc_bins_2[j] + dvcirc_bins_2[j]) for j in range(0, len(bins)-1)]
dvcirc_bins_2_1m = [(vcirc_bins_2[j] - dvcirc_bins_2[j]) for j in range(0, len(bins)-1)]
dvcirc_bins_3_1p = [(vcirc_bins_3[j] + dvcirc_bins_3[j]) for j in range(0, len(bins)-1)]
dvcirc_bins_3_1m = [(vcirc_bins_3[j] - dvcirc_bins_3[j]) for j in range(0, len(bins)-1)]



## chi^2 test ##
def chi2(obs, exp, dobs=None, dexp=None) :
	chi2_tmp = 0
	if len(obs) == len(exp) :
		for i in range(0, len(obs)) :
			if exp[i] != 0 or obs[i] != 0 :
				if dobs is None and dexp is None :
					chi2_i = (obs[i] - exp[i])**2 / exp[i]
				elif dobs is not None and dexp is None :
					chi2_i = ( (obs[i] - exp[i]) / dobs[i] )**2
				elif dobs is None and dexp is not None :
					chi2_i = ( (obs[i] - exp[i]) / dexp[i] )**2
				else :
					chi2_i = (obs[i] - exp[i])**2 / (dobs[i]**2 + dexp[i]**2)
			else :
				chi2_i = 0
			chi2_tmp += chi2_i
	return chi2_tmp #/ len(obs)

def chi2_red(obs, exp, dobs=None, dexp=None) :
	chi2_tmp = 0
	if len(obs) == len(exp) :
		for i in range(0, len(obs)) :
			if exp[i] != 0 or obs[i] != 0 :
				if dobs is None and dexp is None :
					chi2_i = (obs[i] - exp[i])**2 / exp[i]
				elif dobs is not None and dexp is None :
					chi2_i = ( (obs[i] - exp[i]) / dobs[i] )**2
				elif dobs is None and dexp is not None :
					chi2_i = ( (obs[i] - exp[i]) / dexp[i] )**2
				else :
					chi2_i = (obs[i] - exp[i])**2 / (dobs[i]**2 + dexp[i]**2)
			else :
				chi2_i = 0
			chi2_tmp += chi2_i
	return chi2_tmp / len(obs)


# SIDM #
vcirc_SIDM_interp = interpolate.InterpolatedUnivariateSpline(x=r_SIDM, y=vcirc_SIDM, k=3)
vcirc_SIDM_list_tot = vcirc_SIDM_interp(r_exp_tot)
vcirc_SIDM_list_dm = vcirc_SIDM_interp(r_exp_dm)

chi2_SIDM_tot = chi2(obs=vcirc_SIDM_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_SIDM_dm = chi2(obs=vcirc_SIDM_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\n\nchi2_SIDM_tot = ", chi2_SIDM_tot
print "chi2_SIDM_dm = ", chi2_SIDM_dm

# 26 MeV #
# Jim #
vcirc_26MeV_interp = interpolate.InterpolatedUnivariateSpline(x=r_26MeV, y=vcirc_26MeV, k=3)
vcirc_26MeV_list_tot = vcirc_26MeV_interp(r_exp_tot)
vcirc_26MeV_list_dm = vcirc_26MeV_interp(r_exp_dm)

chi2_26MeV_tot = chi2(obs=vcirc_26MeV_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_26MeV_dm = chi2(obs=vcirc_26MeV_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\n\nchi2_26MeV_tot = ", chi2_26MeV_tot, " (Jim)"
print "chi2_26MeV_dm = ", chi2_26MeV_dm, " (Jim)"

# 34 MeV #
# Jim #
vcirc_34MeV_interp = interpolate.InterpolatedUnivariateSpline(x=r_34MeV, y=vcirc_34MeV, k=3)
vcirc_34MeV_list_tot = vcirc_34MeV_interp(r_exp_tot)
vcirc_34MeV_list_dm = vcirc_34MeV_interp(r_exp_dm)

chi2_34MeV_Jim_tot = chi2(obs=vcirc_34MeV_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_34MeV_Jim_dm = chi2(obs=vcirc_34MeV_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\nchi2_34MeV_tot = ", chi2_34MeV_Jim_tot, " (Jim)"
print "chi2_34MeV_dm = ", chi2_34MeV_Jim_dm, " (Jim)"

# 17 MeV #
# Jim #
vcirc_17MeV_interp = interpolate.InterpolatedUnivariateSpline(x=r_17MeV, y=vcirc_17MeV, k=3)
vcirc_17MeV_list_tot = vcirc_17MeV_interp(r_exp_tot)
vcirc_17MeV_list_dm = vcirc_17MeV_interp(r_exp_dm)

chi2_17MeV_Jim_tot = chi2(obs=vcirc_17MeV_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_17MeV_Jim_dm = chi2(obs=vcirc_17MeV_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\nchi2_17MeV_tot = ", chi2_17MeV_Jim_tot, " (Jim)"
print "chi2_17MeV_dm = ", chi2_17MeV_Jim_dm, " (Jim)"


# 001 #
# Jim #
vcirc_001_interp = interpolate.InterpolatedUnivariateSpline(x=r_001, y=vcirc_001, k=3)
vcirc_001_list_tot = vcirc_001_interp(r_exp_tot)
vcirc_001_list_dm = vcirc_001_interp(r_exp_dm)

chi2_001_Jim_tot = chi2(obs=vcirc_001_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_001_Jim_dm = chi2(obs=vcirc_001_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\n\nchi2_001_tot = ", chi2_001_Jim_tot, " (Jim)"
print "chi2_001_dm = ", chi2_001_Jim_dm, " (Jim)"

# 0015 #
# Jim #
vcirc_0015_interp = interpolate.InterpolatedUnivariateSpline(x=r_0015, y=vcirc_0015, k=3)
vcirc_0015_list_tot = vcirc_0015_interp(r_exp_tot)
vcirc_0015_list_dm = vcirc_0015_interp(r_exp_dm)

chi2_0015_Jim_tot = chi2(obs=vcirc_0015_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_0015_Jim_dm = chi2(obs=vcirc_0015_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\nchi2_0015_tot = ", chi2_0015_Jim_tot, " (Jim)"
print "chi2_0015_dm = ", chi2_0015_Jim_dm, " (Jim)"

# 002 #
# Jim #
vcirc_002_interp = interpolate.InterpolatedUnivariateSpline(x=r_002, y=vcirc_002, k=3)
vcirc_002_list_tot = vcirc_002_interp(r_exp_tot)
vcirc_002_list_dm = vcirc_002_interp(r_exp_dm)

chi2_002_Jim_tot = chi2(obs=vcirc_002_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_002_Jim_dm = chi2(obs=vcirc_002_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\nchi2_002_tot = ", chi2_002_Jim_tot, " (Jim)"
print "chi2_002_dm = ", chi2_002_Jim_dm, " (Jim)"



# bins_1 #
# Matteo #
vcirc_bins_1_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=vcirc_bins_1, k=3)
vcirc_bins_1_list_tot = vcirc_bins_1_interp(r_exp_tot)
vcirc_bins_1_list_dm = vcirc_bins_1_interp(r_exp_dm)
dvcirc_bins_1_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=dvcirc_bins_1, k=3)
dvcirc_bins_1_list_tot = dvcirc_bins_1_interp(r_exp_tot)
dvcirc_bins_1_list_dm = dvcirc_bins_1_interp(r_exp_dm)

chi2_bins_1_tot = chi2(obs=vcirc_bins_1_list_tot, exp=vc_exp_tot, dobs=dvcirc_bins_1_list_tot, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_bins_1_dm = chi2(obs=vcirc_bins_1_list_dm, exp=vc_exp_dm, dobs=dvcirc_bins_1_list_dm, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\n\nchi2_bins_1_tot = ", chi2_bins_1_tot, " (Matteo)"
print "chi2_bins_1_dm = ", chi2_bins_1_dm, " (Matteo)"

# bins_2 #
# Matteo #
vcirc_bins_2_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=vcirc_bins_2, k=3)
vcirc_bins_2_list_tot = vcirc_bins_2_interp(r_exp_tot)
vcirc_bins_2_list_dm = vcirc_bins_2_interp(r_exp_dm)
dvcirc_bins_2_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=dvcirc_bins_2, k=3)
dvcirc_bins_2_list_tot = dvcirc_bins_2_interp(r_exp_tot)
dvcirc_bins_2_list_dm = dvcirc_bins_2_interp(r_exp_dm)

chi2_bins_2_tot = chi2(obs=vcirc_bins_2_list_tot, exp=vc_exp_tot, dobs=dvcirc_bins_2_list_tot, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_bins_2_dm = chi2(obs=vcirc_bins_2_list_dm, exp=vc_exp_dm, dobs=dvcirc_bins_2_list_dm, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\nchi2_bins_2_tot = ", chi2_bins_2_tot, " (Matteo)"
print "chi2_bins_2_dm = ", chi2_bins_2_dm, " (Matteo)"

# bins_3 #
# Matteo #
vcirc_bins_3_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=vcirc_bins_3, k=3)
vcirc_bins_3_list_tot = vcirc_bins_3_interp(r_exp_tot)
vcirc_bins_3_list_dm = vcirc_bins_3_interp(r_exp_dm)
dvcirc_bins_3_interp = interpolate.InterpolatedUnivariateSpline(x=mid_bins, y=dvcirc_bins_3, k=3)
dvcirc_bins_3_list_tot = dvcirc_bins_3_interp(r_exp_tot)
dvcirc_bins_3_list_dm = dvcirc_bins_3_interp(r_exp_dm)

chi2_bins_3_tot = chi2(obs=vcirc_bins_3_list_tot, exp=vc_exp_tot, dobs=dvcirc_bins_3_list_tot, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
chi2_bins_3_dm = chi2(obs=vcirc_bins_3_list_dm, exp=vc_exp_dm, dobs=dvcirc_bins_3_list_dm, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "\nchi2_bins_3_tot = ", chi2_bins_3_tot, " (Matteo)"
print "chi2_bins_3_dm = ", chi2_bins_3_dm, " (Matteo)"



print "\nMsun = ", Msun, " GeV"
print "kpc = ", kpc, " cm"


### save the information in dat files ##
#str_out = "DDO154_vcirc_SIDM.dat"
#file_out = open(str_out, "w+")
#for i in range(0, len(r_SIDM)) :
#	file_out.write("%.15f\t%.15f\n" %(r_SIDM[i], vcirc_SIDM[i]))
#file_out.close()
#sys.exit()



#text1 = r'$r_s = {0:0.1f}$'.format(rs1) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1*1.0e10, 1)) + r' M$_{\odot}$'
# vector #
text1 = r'DDO 154 -- Model 1'
text2 = r'$m_X = 100$ MeV' + '\n' + r'$\alpha^{\prime} = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'
## scalar #
#text1 = r'DDO 154 -- Model 2'
#text2 = r'$m_X = 100$ MeV' + '\n' + r'$m_{\phi} \lesssim 2\,m_{\chi}$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'



##-- Plots --##
## circular velocity ##
fig5 = plt.figure(num='vcirc_vs_r', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$V_{\rm circ} (r)$ [km/s]', fontsize=20)
ax5.set_xscale('linear')
ax5.set_yscale('linear')
ax5.set_xlim(0.222, 8.33)
ax5.set_ylim(4.0, 60.0)
ax5.plot(mid_bins, vcircNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'original NFW')
ax5.plot(mid_bins, vcircH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax5.plot(r_SIDM, vcirc_SIDM, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')
datatot=ax5.errorbar(r_exp_tot, vc_exp_tot, xerr=0, yerr=dvc_exp_tot, c='dimgrey', marker='o', ms=8.0, mec='black', mew=0.3, alpha=1.0, linestyle='None', label=r'Total 1502.01281')
datadm=ax5.errorbar(r_exp_dm, vc_exp_dm, xerr=0, yerr=0, c='white', marker='o', ms=8.0, mec='black', mew=0.75, alpha=1.0, linestyle='None', label=r'DM 1502.01281')

ax5.plot(mid_bins, vcirc_bins_2, color='cyan', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV')
ax5.fill_between(mid_bins, dvcirc_bins_2_1m, dvcirc_bins_2_1p, color ='cyan', alpha=0.3)
ax5.plot(r_34MeV, vcirc_34MeV, color ='blue', linestyle ='-.', lw=2.0)
ax5.plot(mid_bins, vcirc_bins_1, color='violet', linestyle='-', lw=2.0, label=r'$m_V = 26$ MeV')
ax5.fill_between(mid_bins, dvcirc_bins_1_1m, dvcirc_bins_1_1p, color ='violet', alpha=0.3)
ax5.plot(r_26MeV, vcirc_26MeV, color ='darkviolet', linestyle ='-.', lw=2.0)
ax5.plot(mid_bins, vcirc_bins_3, color='lime', linestyle='-', lw=2.0, label=r'$m_V = 17$ MeV')
ax5.fill_between(mid_bins, dvcirc_bins_3_1m, dvcirc_bins_3_1p, color ='lime', alpha=0.3)
ax5.plot(r_17MeV, vcirc_17MeV, color ='green', linestyle ='-.', lw=2.0)
ax5.legend([("red","--", " ", " "), ("black",":", " ", " "), ("orange","-", " ", " "), ("cyan","-", "blue","-."), ("violet","-", "darkviolet","-."), ("lime","-", "green","-."), datatot, datadm], [r'Hernquist', r'original NFW', r'SIDM fit 1611.02716', r'$m_V = 34$ MeV', r'$m_V = 26$ MeV', r'$m_V = 17$ MeV', r'Total 1502.01281', r'DM 1502.01281'], handler_map={tuple: AnyObjectHandler()}, loc='lower right', prop={'size': 18})

#ax5.plot(mid_bins, vcirc_bins_1, color='cyan', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.01$')
#ax5.fill_between(mid_bins, dvcirc_bins_1_1m, dvcirc_bins_1_1p, color ='cyan', alpha=0.3)
#ax5.plot(r_001, vcirc_001, color ='blue', linestyle ='-.', lw=2.0)
#ax5.plot(mid_bins, vcirc_bins_2, color='violet', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.015$')
#ax5.fill_between(mid_bins, dvcirc_bins_2_1m, dvcirc_bins_2_1p, color ='violet', alpha=0.3)
#ax5.plot(r_0015, vcirc_0015, color ='darkviolet', linestyle ='-.', lw=2.0)
#ax5.plot(mid_bins, vcirc_bins_3, color='lime', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.02$')
#ax5.fill_between(mid_bins, dvcirc_bins_3_1m, dvcirc_bins_3_1p, color ='lime', alpha=0.3)
#ax5.plot(r_002, vcirc_002, color ='green', linestyle ='-.', lw=2.0)
#ax5.legend([("red","--", " ", " "), ("black",":", " ", " "), ("orange","-", " ", " "), ("cyan","-", "blue","-."), ("violet","-", "darkviolet","-."), ("lime","-", "green","-."), datatot, datadm], [r'Hernquist', r'original NFW', r'SIDM fit 1611.02716', r'$\alpha^{\prime} = 0.01$', r'$\alpha^{\prime} = 0.015$', r'$\alpha^{\prime} = 0.02$', r'Total 1502.01281', r'DM 1502.01281'], handler_map={tuple: AnyObjectHandler()}, loc='lower right', prop={'size': 18})

ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
ax5.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.2))
ax5.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10.0))
ax5.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.0))
ob5 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
ob5_1 = offsetbox.AnchoredText(s=text1, loc='upper left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()
fig5.savefig('test3/figs/vcirc_vs_r_DDO154_vector.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig5.savefig('test3/figs/vcirc_vs_r_DDO154_scalar.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



raw_input()
