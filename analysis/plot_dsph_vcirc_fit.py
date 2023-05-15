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



##-- NFW --##
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

def vcircNFW(r, rhoss, rs) :
	if rhoss >= 0 and rs >= 0 :
		return np.sqrt( (G / r) * 4.0 * np.pi * rhoss * rs**3 * (np.log(1.0 + r / rs) - (r / rs) / (1 + r / rs)) )
	else :
		return 0


# alternative definition of vcircNFW found in eq. (7) in https://arxiv.org/pdf/1502.01281.pdf #
def vcircNFW_alt(r, c, r200) :
	if c >= 0 and r200 >= 0 :
		return np.sqrt(4.0 / 3.0 * np.pi * G * r200**2 * vbar * rhocrit) * np.sqrt( (np.log(1.0 + c * r/r200) - c * r/r200 / (1.0 + c * r/r200)) / (r/r200 * (np.log(1.0 + c) - c / (1.0 + c))) )
	else :
		return 0


# alternative definition #
def vcircNFW_alt_alt(r, c, rs) :
	if c >= 0 and rs >= 0 :
		return np.sqrt( (G / r) * 4.0 / 3.0 * np.pi * (c * rs)**3 * vbar * rhocrit * (np.log(1.0 + r/rs) - r/rs / (1.0 + r/rs)) / (np.log(1.0 + c) - c / (1.0 + c)) )
	else :
		return 0



## Hernquist and simulation parameters in code units!

#------# (real simulation: DDO 154) #-------#
Mv1 = 2.3 # * 1e10 Msun (total mass -> dSph galaxy)
rv1 = R200(Mv1) # kpc
rhos1 = 1.5e-3 # * 1e10 Msun/kpc^3
rs1 = 3.4 # kpc
Vv1 = 49.0 # km/s
c1 = 12.2
eps = 0.3 # kpc
# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
## using inferred M,a from rhos1 and rs1
infile = "test3/hernquist/hernquist_v1_S1"
M = 1.226850076349031
a = 6.187578545092555

sidm_tot = "../../DDO154_sidm_tot.dat"
sidm_dm = "../../DDO154_sidm_dm.dat"
data_exp_tot = "../../DDO154_data_exp_tot.dat"
data_exp_dm = "../../DDO154_data_exp_dm.dat"

logr_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 0)
logrho_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 1)



###---------# (DDO 126 dSph galaxy) #---------##
#Mv1 = 1.02 # * 1e10 Msun (total mass -> dSph galaxy)
#rv1 = R200(Mv1) # kpc
#rhos1 = 1.138e-3 # * 1e10 Msun/kpc^3
#rs1 = 3.455 # kpc
#Vv1 = np.sqrt(G * Mv1 / rv1) # km/s
#c1 = 13.29
#eps = 0.23 # kpc
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




# Hernquist correction
Xmax = 0.99
M /= Xmax # the mass is a bit larger if we want that M(<r_max) = mass we want


## Plot v(r) vs r (similar to what we will do for density rho(r) vs r, radial shells)
numbins = 30
r_min = 0.05
r_max = 10.0
bins = np.linspace(start=r_min, stop=r_max, num=numbins) # left edges


mid_bins = []
dmid_bins = []
for j in range(0, len(bins)-1) :
	hwidth_j = 0.5 * (bins[j+1] - bins[j])
	mid_bins.append(bins[j] + hwidth_j)
	dmid_bins.append(hwidth_j)



# NFW
vcircNFW_th = [vcircNFW(r=mid_bins[j], rhoss=rhos1, rs=rs1) for j in range(0, len(bins)-1)]

# Hernquist
vcircH_th = [vcircH(r=mid_bins[j], M=M, a=a) for j in range(0, len(bins)-1)]


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
print "\nchi2_SIDM_tot = ", chi2_SIDM_tot


vcirc_SIDM_dm_interp = interpolate.InterpolatedUnivariateSpline(x=r_SIDM_dm, y=vcirc_SIDM_dm, k=3)
vcirc_SIDM_dm_list_dm = vcirc_SIDM_dm_interp(r_exp_dm)

chi2_SIDM_dm = chi2(obs=vcirc_SIDM_dm_list_dm, exp=vc_exp_dm, dobs=None, dexp=dvc_exp_tot)#dexp=None) # num point = len(r_exp_dm)
print "chi2_SIDM_dm = ", chi2_SIDM_dm




# alternative fit #
coeffvcircNFW_alt = curve_fit(f=lambda r, c, r200: vcircNFW_alt(r, c, r200), xdata=r_exp_dm, ydata=vc_exp_dm, p0=[c1, rv1], sigma=dvc_exp_tot)
print "\n\nfitted [c, r200] = ", coeffvcircNFW_alt[0]
c_fit_alt = coeffvcircNFW_alt[0][0]
r200_fit_alt = coeffvcircNFW_alt[0][1]
fit_vcircNFW_alt = [vcircNFW_alt(el, c_fit_alt, r200_fit_alt) for el in mid_bins]
print "c1 = ", c1, " rv1 = ", rv1, "\n"

# derived quantities from alternative fit #
rs_fit_alt = r200_fit_alt/c_fit_alt
rhoss_fit_alt = vbar * c_fit_alt**3 / 3.0 * rhocrit * gc(c_fit_alt)
m200_fit_alt = M200s(rhoss_fit_alt, rs_fit_alt, c_fit_alt) # also if we use M200(r200_fit_alt) we get the same result
v200_fit_alt = np.sqrt(G * m200_fit_alt / r200_fit_alt)
print "rs_fit_alt = ", rs_fit_alt
print "rhoss_fit_alt = ", rhoss_fit_alt 
print "m200_fit = ", m200_fit_alt
print "v200_fit = ", v200_fit_alt, "\n"



## main fit ##
coeffvcircNFW = curve_fit(f=lambda r, rhoss, rs: vcircNFW(r, rhoss, rs), xdata=r_exp_dm, ydata=vc_exp_dm, p0=[rhoss_fit_alt, rs_fit_alt], sigma=dvc_exp_tot)
print "\nfitted [rhoss, rs] = ", coeffvcircNFW[0]
rhoss_fit = coeffvcircNFW[0][0]
rs_fit = coeffvcircNFW[0][1]
fit_vcircNFW = [vcircNFW(el, rhoss_fit, rs_fit) for el in mid_bins]
print "rhos1 = ", rhos1, " rs1 = ", rs1, "\n"



# another alternative #
coeffvcircNFW_alt_alt = curve_fit(f=lambda r, c, rs: vcircNFW_alt_alt(r, c, rs), xdata=r_exp_dm, ydata=vc_exp_dm, p0=[c_fit_alt, rs_fit_alt], sigma=dvc_exp_tot)
print "\nfitted [c, rs] = ", coeffvcircNFW_alt_alt[0]
c_fit_alt_alt = coeffvcircNFW_alt_alt[0][0]
rs_fit_alt_alt = coeffvcircNFW_alt_alt[0][1]
fit_vcircNFW_alt_alt = [vcircNFW_alt_alt(el, c_fit_alt_alt, rs_fit_alt_alt) for el in mid_bins]
print "c1 = ", c1, " rs1 = ", rs1, "\n"

# derived quantities from the another alternative fit #
r200_fit_alt_alt = c_fit_alt_alt * rs_fit_alt_alt
rhoss_fit_alt_alt = vbar * c_fit_alt_alt**3 / 3.0 * rhocrit * gc(c_fit_alt_alt)
m200_fit_alt_alt = M200s(rhoss_fit_alt_alt, rs_fit_alt_alt, c_fit_alt_alt) # also if we use M200(r200_fit_alt_alt) we get the same result
v200_fit_alt_alt = np.sqrt(G * m200_fit_alt_alt / r200_fit_alt_alt)
print "r200_fit_alt_alt = ", r200_fit_alt_alt
print "rhoss_fit_alt_alt = ", rhoss_fit_alt_alt 
print "m200_fit = ", m200_fit_alt_alt
print "v200_fit = ", v200_fit_alt_alt




# Best-fit values #
print "\n\n--> FINAL parameters <--"
mv_mean = (m200_fit_alt + m200_fit_alt_alt) / 2.0
rv_mean = (r200_fit_alt + r200_fit_alt_alt) / 2.0
vv_mean = (v200_fit_alt + v200_fit_alt_alt) / 2.0
c_mean = (c_fit_alt + c_fit_alt_alt) / 2.0
rhos_mean = (rhoss_fit + rhoss_fit_alt + rhoss_fit_alt_alt) / 3.0
rs_mean = (rs_fit + rs_fit_alt + rs_fit_alt_alt) / 3.0
print "mv_mean = ", mv_mean, "   Mv1 = ", Mv1
print "rv_mean = ", rv_mean, "   rv1 = ", rv1
print "vv_mean = ", vv_mean, "   Vv1 = ", Vv1
print "c_mean = ", c_mean, "   c1 = ", c1
print "rhos_mean = ", rhos_mean, "   rhos1 = ", rhos1
print "rs_mean = ", rs_mean, "   rs1 = ", rs1
fit_vcircNFW_mean = [vcircNFW(el, rhos_mean, rs_mean) for el in mid_bins]




# Density #
rhoH_th = [rhoH(r=mid_bins[j], M=M*1.0e10, a=a) for j in range(0, len(bins)-1)]
rhoNFW_th = [rhoNFW(r=mid_bins[j], rhoss=rhos1*1.0e10, rs=rs1) for j in range(0, len(bins)-1)]
rhoNFW_fit = [rhoNFW(r=mid_bins[j], rhoss=rhos_mean*1.0e10, rs=rs_mean) for j in range(0, len(bins)-1)]

logmid_bins = [np.log10(el) for el in mid_bins]
logrhoH_th = [np.log10(el * convfac) for el in rhoH_th]
logrhoNFW_th = [np.log10(el * convfac) for el in rhoNFW_th]
logrhoNFW_fit = [np.log10(el * convfac) for el in rhoNFW_fit]


print "\n\nMsun = ", Msun, " GeV"
print "kpc = ", kpc, " cm"


##-- Plots --##
text1 = r'Best-fit params:' + '\n' + r'$r_s = {0:0.1f}$'.format(rs_mean) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos_mean*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$V_v = {0:0.0f}$'.format(vv_mean) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c_mean) + '\n' + r'$M_v = {0:s}$'.format(as_si(mv_mean*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv_mean) + r' kpc'

text2 = r'DDO 154'
#text2 = r'DDO 126'


## circular velocity ##
fig5 = plt.figure(num='vcirc_vs_r_NFWbestfit', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$V_{\rm circ} (r)$ [km/s]', fontsize=20)
ax5.set_xscale('linear')
ax5.set_yscale('linear')

#ax5.set_xlim(0.222, 4.1)
#ax5.set_ylim(0.0, 45.0)
ax5.set_xlim(0.222, 8.33)
ax5.set_ylim(4.0, 60.0)

ax5.plot(mid_bins, vcircH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax5.plot(mid_bins, vcircNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'original NFW')
ax5.plot(r_SIDM_tot, vcirc_SIDM_tot, color ='orange', linestyle ='-', lw=3.0, label=r'Total fit 1611.02716')
ax5.plot(r_SIDM_dm, vcirc_SIDM_dm, color ='blue', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')

datatot=ax5.errorbar(r_exp_tot, vc_exp_tot, xerr=0, yerr=dvc_exp_tot, c='dimgrey', marker='o', ms=8.0, mec='black', mew=0.3, alpha=1.0, linestyle='None', label=r'Total 1502.01281')
datadm=ax5.errorbar(r_exp_dm, vc_exp_dm, xerr=0, yerr=0, c='white', marker='o', ms=8.0, mec='black', mew=0.75, alpha=1.0, linestyle='None', label=r'DM 1502.01281')

ax5.plot(mid_bins, fit_vcircNFW_mean, color ='green', linestyle = '-', lw=2.5, label=r'NFW best fit')
#ax5.plot(mid_bins, fit_vcircNFW, color ='green', linestyle = '-', lw=2.5, label=r'NFW best fit')
#ax5.plot(mid_bins, fit_vcircNFW_alt, color ='lightgreen', linestyle = '--', lw=3.0, label=r'NFW alt best fit')
#ax5.plot(mid_bins, fit_vcircNFW_alt_alt, color ='gray', linestyle = ':', lw=3.0, label=r'NFW alt alt best fit')

ax5.legend(loc='lower right', prop={'size': 18})
ax5.grid(False)
ob5_1 = offsetbox.AnchoredText(s=text1, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
ob5 = offsetbox.AnchoredText(s=text2, loc='upper left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
ax5.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.2))
ax5.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10.0))
ax5.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.0))
fig5.tight_layout()
fig5.show()




# density #
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
#ax4_1.plot(logr_SIDM, logrho_SIDM, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')

ax4_1.plot(logmid_bins, logrhoNFW_fit, color ='green', linestyle ='-', lw=2.5, label=r'NFW best fit')

ax4_1.legend(loc='upper right', prop={'size': 18})
ax4_1.grid(False)
ax4_1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4_1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4_1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4_1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4_1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4_1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax4_1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4_1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
fig4_1.tight_layout()
fig4_1.show()






raw_input()
