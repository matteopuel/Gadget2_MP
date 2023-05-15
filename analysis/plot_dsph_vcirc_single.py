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
Mv1 = 1.23 # * 1e10 Msun (total mass -> dSph galaxy)
rv1 = 41.5 # kpc
rhos1 = 1.5e-3 # * 1e10 Msun/kpc^3
rs1 = 3.4 # kpc
Vv1 = 35.7 # km/s
c1 = 12.2
eps = 0.3 # kpc

# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
## using inferred M,a from rhos1 and rs1
#infile = "test3/hernquist/hernquist_v1_S1"
M = 1.226850076349031
a = 6.187578545092555

sidm_tot = "../../DDO154_sidm_tot.dat"
sidm_dm = "../../DDO154_sidm_dm.dat"
data_exp_tot = "../../DDO154_data_exp_tot.dat"
data_exp_dm = "../../DDO154_data_exp_dm.dat"
data_exp_gas = "../../DDO154_data_exp_gas.dat"
data_exp_star = "../../DDO154_data_exp_star.dat"

## SIDM #
#infile_1 = "test3/hernquist/SIDM/DDO154/out/snp_013"

# VECTOR #
mV_text = r'$m_V = 27$ MeV'
infile_1 = "test3/hernquist/vector/DDO154_0015_27/out/snp_013"




###---------# (DDO 126 dSph galaxy) #---------##
#Mv1 = 1.02 # * 1e10 Msun (total mass -> dSph galaxy)
#rv1 = R200(Mv1) # kpc
#rhos1 = 1.138e-3 # * 1e10 Msun/kpc^3
#rs1 = 3.455 # kpc
#Vv1 = np.sqrt(G * Mv1 / rv1) # km/s
#c1 = 13.29
#eps = 0.23 # kpc
#
## Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
### using inferred M, a from rhos1 and rs1
#infile = "test3/hernquist/hernquist_v1_S1_ddo126"
#M = 1.0200621668624161
#a = 6.425815163925038
#
#sidm_tot = "../../DDO126_sidm_tot.dat"
#sidm_dm = "../../DDO126_sidm_dm.dat"
#data_exp_tot = "../../DDO126_data_exp_tot.dat"
#data_exp_dm = "../../DDO126_data_exp_dm.dat"
#data_exp_gas = "../../DDO126_data_exp_gas.dat"
#data_exp_star = "../../DDO126_data_exp_star.dat"
#
## SIDM #
#mV_text = r'$m_V = 34$ MeV'
#infile_1 = "test3/hernquist/SIDM/DDO126/out/snp_013"
##
#### VECTOR #
###mV_text = r'$m_V = 34$ MeV'
###infile_1 = "test3/hernquist/vector/DDO126_34/out/snp_013"
##
#### SCALAR #
###mV_text = r'$\alpha\prime = 0.01$'
###infile_1 = "test3/hernquist/scalar/DDO126_benchmark/out/snp_013"
###mV_text = r'$\alpha\prime = 0.015$'
###infile_1 = "test3/hernquist/scalar/DDO126_0015/out/snp_013"
###mV_text = r'$\alpha\prime = 0.02$'
###infile_1 = "test3/hernquist/scalar/DDO126_002/out/snp_013"


## SIDM in https://arxiv.org/pdf/1611.02716.pdf #
r_SIDM_tot = np.loadtxt(fname=sidm_tot, delimiter=' ', usecols = 0)
vcirc_SIDM_tot = np.loadtxt(fname=sidm_tot, delimiter=' ', usecols = 1)
r_SIDM_dm = np.loadtxt(fname=sidm_dm, delimiter=' ', usecols = 0)
vcirc_SIDM_dm = np.loadtxt(fname=sidm_dm, delimiter=' ', usecols = 1)


## REAL DATA ##
r_exp_tmp = np.loadtxt(fname=data_exp_tot, delimiter=' ', usecols = 0)
vc_exp_tmp = np.loadtxt(fname=data_exp_tot, delimiter=' ', usecols = 1)
r_exp_dm = np.loadtxt(fname=data_exp_dm, delimiter=' ', usecols = 0)
vc_exp_dm = np.loadtxt(fname=data_exp_dm, delimiter=' ', usecols = 1)
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


r_exp_gas = np.loadtxt(fname=data_exp_gas, delimiter=' ', usecols = 0)
vcirc_exp_gas = np.loadtxt(fname=data_exp_gas, delimiter=' ', usecols = 1)
r_exp_star = np.loadtxt(fname=data_exp_star, delimiter=' ', usecols = 0)
vcirc_exp_star = np.loadtxt(fname=data_exp_star, delimiter=' ', usecols = 1)



# for density #
logr_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 0)
logrho_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 1)




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






##-- Set the IC file --##
Ntot_1 = readGadget1.readHeader(filename=infile_1, strname='npart')
Mtot = readGadget1.readHeader(filename=infile_1, strname='mass')
time = readGadget1.readHeader(filename=infile_1, strname='time')

print "Ntot = ", Ntot_1
print "\nMtot = ", Mtot
print "time = ", time, "\n"

mp = Mtot[1] # 1e10 Msun


# Hernquist correction
Xmax = 0.99
M /= Xmax # the mass is a bit larger if we want that M(<r_max) = mass we want


##-- Read the particle properties --##
PPosDM_1, PVelDM_1, _, PMassDM_1 = readGadget1.readSnapshot(filename=infile_1, ptype='dm', strname='full', full=True, mass=False)


##-- Velocity distribution --##
# (Maxwell-Boltzmann? NO, maybe because f(0.5 * v**2 - G * M / (r + a)) is more complicated)
x_1, y_1, z_1 = getCoord(vec=PPosDM_1)
pos_r_1 = getRad(x=x_1, y=y_1, z=z_1)
vel_x_1, vel_y_1, vel_z_1 = getCoord(vec=PVelDM_1)
vel_tot_1 = getRad(x=vel_x_1, y=vel_y_1, z=vel_z_1)

phi_1 = [np.arctan(y_1[i] / x_1[i]) for i in range(0, len(pos_r_1))] 
theta_1 = [np.arccos(z_1[i] / pos_r_1[i]) for i in range(0, len(pos_r_1))]

# my method
vel_r_1 = [vel_x_1[i]*np.sin(theta_1[i])*np.cos(phi_1[i]) + vel_y_1[i]*np.sin(theta_1[i])*np.sin(phi_1[i]) + vel_z_1[i]*np.cos(theta_1[i]) for i in range(0, len(vel_tot_1))]
vel_theta_1 = [vel_x_1[i]*np.cos(theta_1[i])*np.cos(phi_1[i]) + vel_y_1[i]*np.cos(theta_1[i])*np.sin(phi_1[i]) - vel_z_1[i]*np.sin(theta_1[i]) for i in range(0, len(vel_tot_1))]
vel_phi_1 = [-vel_x_1[i]*np.sin(phi_1[i]) + vel_y_1[i]*np.cos(phi_1[i]) for i in range(0, len(vel_tot_1))] 



## Plot v(r) vs r (similar to what we will do for density rho(r) vs r, radial shells)
numbins = 30
mp *= 1.0e10 # Msun

r_min_1 = np.around(a=min(pos_r_1), decimals=1)
r_max_1 = np.around(a=max(pos_r_1), decimals=0) + 1.0
print "\nr_min = ", r_min_1, " kpc"
print "r_max = ", r_max_1, " kpc"


r_max = 100.0 * a
if r_min_1 == 0.0 :
	r_min = 0.05
else :
	r_min = 0.05 # kpc


#bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # left edges
r_max = 10.0
#bins = np.linspace(start=r_min, stop=r_max, num=numbins) # left edges
bins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins) # left edges


Npart_bins_1 = [0 for j in range(0, len(bins)-1)]
Mp_bins_1 = [[] for j in range(0, len(bins)-1)]
vel_bins_1 = [[] for j in range(0, len(bins)-1)]
pos_bins_1 = [[] for j in range(0, len(bins)-1)]
for i in range(0, len(pos_r_1)) :	
	for j in range(0, len(bins)-1) :
		if pos_r_1[i] >= bins[j] and pos_r_1[i] < bins[j+1] :
			Npart_bins_1[j] += 1
			Mp_bins_1[j].append(PMassDM_1[i])
			vel_bins_1[j].append(vel_r_1[i])
			pos_bins_1[j].append(pos_r_1[i])
			break

dNpart_bins_1 = [np.sqrt(el) for el in Npart_bins_1]


mid_bins = []
dmid_bins = []
Menc_bins_1 = []
dMenc_bins_1 = []
menc_1 = 0
dmenc2_1 = 0
vel_mean_bins_1 = []
dvel_mean_bins_1 = []
rho_bins_1 = []
drho_bins_1 = []
for j in range(0, len(bins)-1) :
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)
	vel_sum_j = sum(vel_bins_1[j]) # len(vel_bins[j]) == Npart_bins[j]
	if (Npart_bins_1[j] != 0) :
		mass_j = sum(Mp_bins_1[j])
		dmass_j = np.sqrt(Npart_bins_1[j]) * (mass_j / Npart_bins_1[j]) # Poisson uncertainty of N counts
		menc_1 += mass_j
		dmenc2_1 += dmass_j**2
		vel_mean_bins_1.append(vel_sum_j/Npart_bins_1[j])
		dvel_mean_bins_1.append(vel_sum_j/(Npart_bins_1[j]**1.5))
		vol_j = 4.0 * np.pi / 3.0 * (bins[j+1]**3 - bins[j]**3)
		#rho_bins_1.append(mass_j / vol_j) # does not work!
		#drho_bins_1.append(dmass_j / vol_j) # does not work!
		rho_bins_1.append(Npart_bins_1[j] * mp / vol_j)
		drho_bins_1.append(np.sqrt(Npart_bins_1[j]) * mp / vol_j)
	else :
		menc_1 += 0.0
		dmenc2_1 += 0.0
		vel_mean_bins_1.append(0.0)
		dvel_mean_bins_1.append(0.0)
		rho_bins_1.append(0.0)
		drho_bins_1.append(0.0)

	Menc_bins_1.append(menc_1)
	dMenc_bins_1.append(np.sqrt(dmenc2_1))


vcirc_bins_1 = []
dvcirc_bins_1 = []
for j in range(0, len(bins)-1) :
	vc_j_1 = np.sqrt(G * Menc_bins_1[j] / mid_bins[j])
	dvc_j_1 = vc_j_1 / 2.0 * np.sqrt((dMenc_bins_1[j] / Menc_bins_1[j])**2 + (dmid_bins[j] / mid_bins[j])**2)
	vcirc_bins_1.append(vc_j_1)
	dvcirc_bins_1.append(dvc_j_1)


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



# NFW
vcircNFW_th = [vcircNFW(r=mid_bins[j], rhoss=rhos1, rs=rs1, c=c1) for j in range(0, len(bins)-1)]
#rhoNFW_th = [rhos(s=mid_bins[j]/rv1, c=c1, Mv=Mv1*1.0e10, rv=rv1) for j in range(0, len(bins)-1)]
rhoNFW_th = [rhoNFW(r=mid_bins[j], rhoss=rhos1*1.0e10, rs=rs1) for j in range(0, len(bins)-1)]
veldispNFW_th = [veldisps(s=mid_bins[j]/(c1 * rs1), c=c1, Vv=Vv1) for j in range(0, len(bins)-1)]

# Hernquist
vcircH_th = [vcircH(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)]
rhoH_th = [rhoH(r=mid_bins[j], M=M*1.0e10, a=a) for j in range(0, len(bins)-1)]
veldispH_th = [veldispH(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)] # G*M has right units #/np.sqrt(3.0) would work!
vescH = [vesc(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)]


dvcirc_bins_1_1p = [(vcirc_bins_1[j] + dvcirc_bins_1[j]) for j in range(0, len(bins)-1)]
dvcirc_bins_1_1m = [(vcirc_bins_1[j] - dvcirc_bins_1[j]) for j in range(0, len(bins)-1)]
drho_bins_1_1p = [(rho_bins_1[j] + drho_bins_1[j]) for j in range(0, len(bins)-1)]
drho_bins_1_1m = [(rho_bins_1[j] - drho_bins_1[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_1_1p = [(vel_std_bins_1[j] + dvel_std_bins_1[j]) for j in range(0, len(bins)-1)]
dvel_std_bins_1_1m = [(vel_std_bins_1[j] - dvel_std_bins_1[j]) for j in range(0, len(bins)-1)]


logmid_bins = [np.log10(el) for el in mid_bins]

logrho_bins_1 = [np.log10(el * convfac) if el > 0 else -2.0 for el in rho_bins_1]
logdrho_bins_1_1p = [np.log10(drho_bins_1_1p[j] * convfac) if drho_bins_1_1p[j] != 0 else 0 for j in range(0, len(bins)-1)]
logdrho_bins_1_1m = [np.log10(drho_bins_1_1m[j] * convfac) if drho_bins_1_1m[j] != 0 else 0 for j in range(0, len(bins)-1)]
logrhoH_th = [np.log10(el * convfac) for el in rhoH_th]
logrhoNFW_th = [np.log10(el * convfac) for el in rhoNFW_th]


# SIDM curve conversion #
r_SIDM = [10.0**el for el in logr_SIDM]
rho_SIDM = [(10.0**el) / convfac for el in logrho_SIDM]




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
vcirc_SIDM_tot_interp = interpolate.InterpolatedUnivariateSpline(x=r_SIDM_tot, y=vcirc_SIDM_tot, k=3)
vcirc_SIDM_tot_list_tot = vcirc_SIDM_tot_interp(r_exp_tot)

chi2_SIDM_tot = chi2(obs=vcirc_SIDM_tot_list_tot, exp=vc_exp_tot, dobs=None, dexp=dvc_exp_tot) # num point = len(r_exp_tot)
print "\n\nchi2_SIDM_tot = ", chi2_SIDM_tot


vcirc_SIDM_dm_interp = interpolate.InterpolatedUnivariateSpline(x=r_SIDM_dm, y=vcirc_SIDM_dm, k=3)
vcirc_SIDM_dm_list_dm = vcirc_SIDM_dm_interp(r_exp_dm)

chi2_SIDM_dm = chi2(obs=vcirc_SIDM_dm_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "chi2_SIDM_dm = ", chi2_SIDM_dm



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



# Subtraction of gas and star contributions #
vcirc_exp_gas_interp = interpolate.InterpolatedUnivariateSpline(x=r_exp_gas, y=vcirc_exp_gas, k=3)
vcirc_exp_gas_list = vcirc_exp_gas_interp(r_exp_dm)
vcirc_exp_star_interp = interpolate.InterpolatedUnivariateSpline(x=r_exp_star, y=vcirc_exp_star, k=3)
vcirc_exp_star_list = vcirc_exp_star_interp(r_exp_dm)
vcirc_bins_1_list_dm_res = [np.sqrt(vcirc_bins_1_list_dm[j]**2 - vcirc_exp_gas_list[j]**2 - vcirc_exp_star_list[j]**2) for j in range(0, len(r_exp_dm))]
dvcirc_bins_1_list_dm_res_1p = [(vcirc_bins_1_list_dm_res[j] + dvcirc_bins_1_list_dm[j]) for j in range(0, len(r_exp_dm))]
dvcirc_bins_1_list_dm_res_1m = [(vcirc_bins_1_list_dm_res[j] - dvcirc_bins_1_list_dm[j]) for j in range(0, len(r_exp_dm))]



print "\nMsun = ", Msun, " GeV"
print "kpc = ", kpc, " cm"


### save the information in dat files ##
#str_out = "DDO154_vcirc_SIDM.dat"
#file_out = open(str_out, "w+")
#for i in range(0, len(r_SIDM)) :
#	file_out.write("%.15f\t%.15f\n" %(r_SIDM[i], vcirc_SIDM[i]))
#file_out.close()
#sys.exit()



#galname = r'DDO 126'
galname = r'DDO 154'
#text1 = galname

# vector #
text1 = galname + r' -- Model 1'
#text2 = r'$m_X = 100$ MeV' + '\n' + r'$\alpha\prime = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'
text2 = r'$m_X = 100$ MeV' + '\n' + r'$\alpha\prime = 0.015$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'

## scalar #
#text1 = galname + r' -- Model 2'
#text2 = r'$m_X = 100$ MeV' + '\n' + r'$m_{\phi} = 70$ MeV' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


text3 = r'$r_s = {0:0.1f}$'.format(rs1) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$V_v = {0:0.0f}$'.format(Vv1) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1) + r' kpc'


##-- Plots --##
fig3 = plt.figure(num='stdv_vs_r', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax3.set_ylabel(r'$\sigma_{v_r} (r)$ [km/s]', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(0.085, r_max)
ax3.set_ylim(1.0e0, 1.0e2)
ax3.plot(mid_bins, veldispH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax3.plot(mid_bins, veldispNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax3.plot(mid_bins, vel_std_bins_1, color='darkviolet', linestyle='-', lw=2.0, label=mV_text)
ax3.fill_between(mid_bins, dvel_std_bins_1_1m, dvel_std_bins_1_1p, color ='darkviolet', alpha=0.3)
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='lower right', prop={'size': 18})
ob3 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
ob3_1 = offsetbox.AnchoredText(s=text1, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob3_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3_1)
ob3_11 = offsetbox.AnchoredText(s=text3, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob3_11.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3_11)
fig3.tight_layout()
fig3.show()



fig4 = plt.figure(num='rho_vs_r', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax4.set_ylabel(r'$\rho (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(0.1, r_max)
ax4.set_ylim(1.0e2, 1.0e10)
ax4.plot(mid_bins, rhoH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax4.plot(mid_bins, rhoNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax4.plot(r_SIDM, rho_SIDM, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')
ax4.plot(mid_bins, rho_bins_1, color='darkviolet', linestyle='-', lw=2.0, label=mV_text)
ax4.fill_between(mid_bins, drho_bins_1_1m, drho_bins_1_1p, color ='darkviolet', alpha=0.3)
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.03, y=1.0e5, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=text1, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
ob4_11 = offsetbox.AnchoredText(s=text3, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4_11.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_11)
fig4.tight_layout()
fig4.show()



fig4_1 = plt.figure(num='logrho_vs_r', figsize=(10, 7), dpi=100)
ax4_1 = fig4_1.add_subplot(111)
ax4_1.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax4_1.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax4_1.set_xscale('linear')
ax4_1.set_yscale('linear')
ax4_1.set_xlim(-1.0, 0.9)
ax4_1.set_ylim(-1.9, 1.4)
ax4_1.plot(logmid_bins, logrhoH_th, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax4_1.plot(logmid_bins, logrhoNFW_th, color ='black', linestyle =':', lw=2.0, label=r'NFW')
ax4_1.plot(logr_SIDM, logrho_SIDM, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')

ax4_1.plot(logmid_bins, logrho_bins_1, color='darkviolet', linestyle='-', lw=2.0, label=mV_text)
ax4_1.fill_between(logmid_bins, logdrho_bins_1_1m, logdrho_bins_1_1p, color ='darkviolet', alpha=0.3)
ax4_1.legend(loc='upper right', prop={'size': 18})
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
ob4_1 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4_1.add_artist(ob4_1)
ob4_1_1 = offsetbox.AnchoredText(s=text1, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4_1_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4_1.add_artist(ob4_1_1)
ob4_1_11 = offsetbox.AnchoredText(s=text3, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=16))
ob4_1_11.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4_1.add_artist(ob4_1_11)
fig4_1.tight_layout()
fig4_1.show()





## circular velocity ##
fig5 = plt.figure(num='vcirc_vs_r', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$V_{\rm circ} (r)$ [km/s]', fontsize=20)
ax5.set_xscale('linear')
ax5.set_yscale('linear')

ax5.set_xlim(0.222, 8.33)
ax5.set_ylim(0.0, 58.0)
#ax5.set_xlim(0.222, 4.1)
#ax5.set_ylim(0.0, 45.0)

ax5.plot(mid_bins, vcircH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax5.plot(mid_bins, vcircNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'original NFW')
ax5.plot(r_SIDM_tot, vcirc_SIDM_tot, color ='orange', linestyle ='-', lw=3.0, label=r'Total fit 1611.02716')
ax5.plot(r_SIDM_dm, vcirc_SIDM_dm, color ='blue', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')
datatot=ax5.errorbar(r_exp_tot, vc_exp_tot, xerr=0, yerr=dvc_exp_tot, c='dimgrey', marker='o', ms=8.0, mec='black', mew=0.3, alpha=1.0, linestyle='None', label=r'Total 1502.01281')
datadm=ax5.errorbar(r_exp_dm, vc_exp_dm, xerr=0, yerr=0, c='white', marker='o', ms=8.0, mec='black', mew=0.75, alpha=1.0, linestyle='None', label=r'DM 1502.01281')

ax5.plot(mid_bins, vcirc_bins_1, color='darkviolet', linestyle='-', lw=2.0, label=mV_text)
ax5.fill_between(mid_bins, dvcirc_bins_1_1m, dvcirc_bins_1_1p, color ='darkviolet', alpha=0.3)

ax5.plot(r_exp_dm, vcirc_bins_1_list_dm_res, color='green', linestyle='-', lw=2.0, label=r'without gas and stars')
ax5.fill_between(r_exp_dm, dvcirc_bins_1_list_dm_res_1m, dvcirc_bins_1_list_dm_res_1p, color ='green', alpha=0.3)
ax5.plot(r_exp_gas, vcirc_exp_gas, color ='magenta', linestyle ='-.', lw=2.0, label=r'gas 1502.01281')
ax5.plot(r_exp_star, vcirc_exp_star, color ='magenta', linestyle ='--', lw=2.0, label=r'star 1502.01281')
ax5.legend(loc='lower right', prop={'size': 18})

ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
ax5.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.2))
ax5.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10.0))
ax5.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.0))
#ob5 = offsetbox.AnchoredText(s=text2, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
#ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
#ax5.add_artist(ob5)
ob5_1 = offsetbox.AnchoredText(s=text1, loc='upper left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()
#fig5.savefig('test3/figs/vcirc_vs_r_DDO126_vector.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



raw_input()
