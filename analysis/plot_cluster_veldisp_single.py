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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from decimal import *
getcontext().prec = 8 # memorize 8 digits after the comma
from scipy import interpolate
from scipy.optimize import curve_fit, minimize

import scipy.special as sc

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


def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r', edgecolor='None', alpha=0.5, label='data'):

    # Loop over data points; create box from errors at each point
    xerrorT = xerror.T
    lowleftpt_x = [xdata[j] - xerrorT[j][0] for j in range(0, len(xdata))]
    lowleftpt_y = [ydata[j] - yerror[j] for j in range(0, len(ydata))]
    err_x = [xerrorT[j][0] + xerrorT[j][1] for j in range(0, len(xerrorT))]
    err_y = [2.0 * yerror[j] for j in range(0, len(yerror))]
    errorboxes = [Rectangle((lowleftpt_x[j], lowleftpt_y[j]), err_x[j], err_y[j]) for j in range(0, len(xdata))]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, fmt='None', ecolor='k', label=label)

    return artists


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


#--> conversion into kpc
def arcsec2kpc(arcs, z, circ=False) :
	cc = 299792.458 # km/s
	h = 0.674
	H0 = h * 100.0 / 1000.0 # km/s/kpc
	arcs2deg = arcs / 3600.0 # deg
	arcs2rad = arcs2deg * np.pi / 180.0 # rad
	res = cc * z * arcs2rad / H0
	if circ == True :
		return res
	else :
		return res / np.sqrt(boa) #/ 2.0 # the factor 1.0/np.sqrt(boa) comes from the circularization and 1.0/2.0 comes from the fact that the subtending angle corresponds to the diameter of the object



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
	arg = np.pi**2 - np.log(x) - 1.0 / x - 1.0 / (1.0 + x)**2 - 6.0 / (1.0 + x) + (1.0 + 1.0 / x**2 - 4.0 / x - 2.0 / (1.0 + x)) * np.log(1.0 + x) + 3.0 * (np.log(1.0 + x))**2 + 6.0 * sc.spence(1.0 + x)
	tmp = 0.5 * c**2 * gc(c) * s * (1.0 + x)**2 * arg
	if tmp < 0 :
		tmp = 0
	return Vv * np.sqrt(tmp)


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
rv1 = 2046.0 # kpc
Mv1 = 127767 # * 1e10 Msun (total mass -> galaxy cluster)
rhos1 = 1.3e-4 # * 1e10 Msun/kpc^3
rs1 = 442.0 # kpc
Vv1 = 1639.0 # km/s
c1 = 4.63

eps = 9.3 # kpc

# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
## using inferred M,a from rhos1 and rs1
infile = "test3/hernquist/hernquist_v1_S1_cluster"
M = 127766.92971544608 # * 1e10 Msun
a = 594.8897476275504 # kpc



## REAL DATA ##
## Import observational data: circularized vs projected stdv ##
cr_data_exp = np.loadtxt(fname='../../A2537_data_exp_tot.dat', delimiter='\t', usecols = 0)
dcr_data_exp_l = np.loadtxt(fname='../../A2537_data_exp_tot.dat', delimiter='\t', usecols = 1)
dcr_data_exp_r = np.loadtxt(fname='../../A2537_data_exp_tot.dat', delimiter='\t', usecols = 2)
proj_stdv_data_exp = np.loadtxt(fname='../../A2537_data_exp_tot.dat', delimiter='\t', usecols = 3)
dproj_stdv_data_exp = np.loadtxt(fname='../../A2537_data_exp_tot.dat', delimiter='\t', usecols = 4)


## from Table 6 in https://arxiv.org/pdf/1209.1391.pdf: radial bins (not-circularized, oriented along major-axis) in arcsec vs projected stdv ##
rb_data_exp_tab_l = [0.0, 0.41, 1.22, 2.03, 2.84]
rb_data_exp_tab_r = [0.41, 1.22, 2.03, 2.84, 3.65]
proj_stdv_data_exp_tab = [284.0, 315.0, 328.0, 360.0, 385.0]
dproj_stdv_data_exp_tab = [14.0, 19.0, 20.0, 22.0, 43.0]
## maybe not used ... ##


## SIDM fit in fig.2 in https://arxiv.org/pdf/1508.03339.pdf ##
logr_SIDM_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 0)
logrho_SIDM_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 1)
logdrho_SIDM_1p_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 2)
logdrho_SIDM_1m_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 3)

r_SIDM_bar_c = np.loadtxt(fname='../../A2537_rho_SIDM_b.dat', delimiter='\t', usecols = 0) # [kpc]
rho_SIDM_bar_c = np.loadtxt(fname='../../A2537_rho_SIDM_b.dat', delimiter='\t', usecols = 1) # [Msun / kpc^3]
drho_SIDM_bar_c = np.loadtxt(fname='../../A2537_rho_SIDM_b.dat', delimiter='\t', usecols = 2) # [Msun / kpc^3]

r_SIDM_fit_cpm = np.loadtxt(fname='../../A2537_veldisplos_SIDM.dat', delimiter='\t', usecols = 0)
vdisp_SIDM_fit_c = np.loadtxt(fname='../../A2537_veldisplos_SIDM.dat', delimiter='\t', usecols = 1)
vdisp_SIDM_fit_p = np.loadtxt(fname='../../A2537_veldisplos_SIDM.dat', delimiter='\t', usecols = 2)
vdisp_SIDM_fit_m = np.loadtxt(fname='../../A2537_veldisplos_SIDM.dat', delimiter='\t', usecols = 3)



## Original parameters for A 2537 (https://arxiv.org/pdf/1209.1391.pdf) ##
boa = 0.74
z_a2537 = 0.294 #0.2966
wslit_arcs = 1.5 # [arcsec] # not circularized
seeing_arcs = 0.8 # [arcsec] # FWHM
rcut = 52.7 # [kpc] # circularized
drcut = 6.5 # [kpc] # circularized
rcore = 0.75 # [kpc]
Mag = 16.9
dMag = 0.1
Lv = 5.86e11 # [Lsun]
# assimpution: YSPS and Ys are spatially-invariant
YSPS = 2.32 # [Msun/Lsun] # Chabrier IMF (V-band)
LogYoYSPS_min = -0.176
LogYoYSPS_max = 0.551


## derived parameters ##
seeing = arcsec2kpc(arcs=seeing_arcs, z=z_a2537, circ=True) # [kpc]
sigmaPSF = seeing / (2.0 * np.sqrt(2.0 * np.log(2.0))) # [kpc] # definition of FWHM
wslit_arcs *= np.sqrt(boa) # [arcsec] # circularized
wslit = arcsec2kpc(arcs=wslit_arcs, z=z_a2537, circ=True) # [kpc]
#YoYSPS_min = 10.0**LogYoYSPS_min
#YoYSPS_max = 10.0**LogYoYSPS_max
#print "YoYSPS_min = ", YoYSPS_min
#print "YoYSPS_max = ", YoYSPS_max



#--> model stellar luminosity density profile
def rho_dPIE(r, rho_0, r_core, r_cut) :
	return rho_0 / (1.0 + (r / r_core)**2) / (1.0 + (r / r_cut)**2) # [Lsun / kpc^3]

def Sigma_dPIE(R, rho_0, r_core, r_cut) :
	return np.pi * rho_0 * (r_core * r_cut)**2 / (r_cut**2 - r_core**2) * (1.0 / np.sqrt(R**2 + r_core**2) - 1.0 / np.sqrt(R**2 + r_cut**2)) # [Lsun / kpc^2]

def L_dPIE(R, rho_0, r_core, r_cut) :
	return 4.0 * np.pi * rho_0 * (r_core * r_cut)**2 / (r_cut**2 - r_core**2) * (r_cut * np.arctan(R / r_cut) - r_core * np.arctan(R / r_core)) # [Lsun]

def M_star(r, YoYSPS, Y_SPS, rho_0, r_core, r_cut) :
	return YoYSPS * Y_SPS * L_dPIE(r, rho_0, r_core, r_cut) # [Msun] # YoYSPS is a free parameter


# rho0 is determined by setting Sigma_dPIE(Rt) = 0, which occurs for Rt = np.inf, leading to Lv = L_dPIE(Rt)
def rho0(L_v, r_core, r_cut) :
	return L_v * (r_cut + r_core) / (2.0 * np.pi**2 * (r_core * r_cut)**2) # [Lsun / kpc^3]


# general defintion of F(r) in https://arxiv.org/pdf/0806.0042.pdf (eq. 43)
def beta_veldisp(sigamr, sigmatheta, sigmaphi) :
	return 1.0 - (sigmatheta**2 + sigmaphi**2) / (2.0 * sigmar)

def Fbeta(r, R, beta_gen) :
	omega = (R / r)**2
	if beta_gen == 0 :
		if r >= R : 
			res = np.sqrt(r**2 - R**2) # [kpc] # if beta_gen = 0 (isotropic vel. disp.)
		else :
			res = 0
	else :
		if r >= R :
			apos = beta_gen + 0.5
			aneg = beta_gen - 0.5
			Bpos = sc.betainc(apos, 0.5, omega) * sc.beta(apos, 0.5)
			Bneg = sc.betainc(aneg, 0.5, omega) * sc.beta(aneg, 0.5)
			beta_gen05 = beta_gen - 0.5
			res = 0.5 * R**(1.0 - 2.0 * beta_gen) * (beta_gen * Bpos - Bneg + sc.gamma(beta_gen05) * np.sqrt(np.pi) * (1.5 - beta_gen) / sc.gamma(beta_gen)) # [kpc^(1-2*beta_gen)]
		else :
			res = 0
	return res



## Set basic quantities ##
numbins = 100
# new axis #
r_list = np.linspace(start=1.0, stop=100.0, num=numbins) # [kpc]


# since R should be R = np.sqrt(x**2 + y**2) and x is in [R_min, R_max]	and y is in [-wslit/2, wslit/2], we can symmetrize the y-integral by multiplying the result by 2 and the integration goes from [0, wslit/2]
ylist = np.linspace(start=0.0, stop=wslit/2.0, num=numbins) # [kpc]
xlist = np.linspace(start=0.0, stop=25.0, num=numbins) # [kpc]
Rlist = [np.sqrt(xlist[j]**2 + ylist[j]**2) for j in range(0, len(xlist))]


# Used later for plot and also now #
bins = np.linspace(start=0.5, stop=25.0, num=int(numbins/2.0))
binsL = bins[:-1]
binsR = bins[1:]
bin_widths = binsL - binsR
mid_bins = binsL + 0.5 * bin_widths


# assumption: beta = 1 - sigma_t**2 / sigma_r**2 = 0 --> isotropic velocity dispersion
betagen = 0 # = beta0 --> use Fbeta0(r, R)
rho0s = rho0(L_v=Lv, r_core=rcore, r_cut=rcut) # [Lsun / kpc^3]


def rhos(r) :
	return rho_dPIE(r, rho_0=rho0s, r_core=rcore, r_cut=rcut) # [Lsun / kpc^3]

def Sigmas(R) :
	return Sigma_dPIE(R, rho_0=rho0s, r_core=rcore, r_cut=rcut) # [Lsun / kpc^2]

def Mstars(r, YoYSPS) :
	return M_star(r, YoYSPS, Y_SPS=YSPS, rho_0=rho0s, r_core=rcore, r_cut=rcut) # [Msun]


def rhob_s(r, YoYSPS, Y_SPS) :
	return YoYSPS * Y_SPS * rhos(r) # [Msun / kpc^3]



## find the best-fit LogYoYSPS from SIDM data ##
#tmp_list = np.linspace(start=min(r_SIDM_bar_c), stop=max(r_SIDM_bar_c), num=1000)
tmp_list = np.linspace(start=min(r_SIDM_bar_c), stop=14.0, num=1000)

rho_SIDM_bar_c_interp = interpolate.InterpolatedUnivariateSpline(x=r_SIDM_bar_c, y=rho_SIDM_bar_c, k=3)
drho_SIDM_bar_c_interp = interpolate.InterpolatedUnivariateSpline(x=r_SIDM_bar_c, y=drho_SIDM_bar_c, k=3)

rho_SIDM_bar_c_tmp = rho_SIDM_bar_c_interp(tmp_list)
drho_SIDM_bar_c_tmp = drho_SIDM_bar_c_interp(tmp_list)
rhob_tmp = [rhob_s(r=el, YoYSPS=1.0, Y_SPS=YSPS) for el in tmp_list]



def chi2(YoYSPSfit) :
	chi2_tmp = np.sum( ((rho_SIDM_bar_c_tmp - YoYSPSfit * rhob_tmp) / drho_SIDM_bar_c_tmp )**2 )
	return chi2_tmp

YoYSPSfit_0 = np.array([1.0])
totres =  minimize(chi2, YoYSPSfit_0)
YoYSPSfit_best = totres.x[0]
LogYoYSPS0 = np.log10(YoYSPSfit_best)


def chi2_best(YoYSPSfit=YoYSPSfit_best) :
	chi2_tmp = 0
	for i in range(0, len(tmp_list)) :
		chi2_i = ( (rho_SIDM_bar_c_tmp[i] - YoYSPSfit * rhob_tmp[i]) / drho_SIDM_bar_c_tmp[i] )**2
		chi2_tmp += chi2_i
	return chi2_tmp


print "chi2 = ", chi2_best(YoYSPSfit=YoYSPSfit_best)
print "YoYSPSfit_best = ", YoYSPSfit_best
print "LogYoYSPS0 = ", LogYoYSPS0


#sys.exit()
#LogYoYSPS0 = np.log10(2.055)


## Relative data ##
#LogYoYSPS0 = 0 # for simplicity, but it can range in [YoYSPS_min, YoYSPS_max]
YoYSPS0 = 10.0**LogYoYSPS0


##--> projected stellar velocity dispersion along the line-of-sight <--##
def veldispLOS(logr, logrho, Rmin, Rmax) :
	
	rDM_tmp = [10.0**el for el in logr] # kpc
	rhoDM_tmp = [(10.0**el) / convfac for el in logrho] # [Msun/kpc^3]
	MrDM_int = [4.0 * np.pi * rDM_tmp[j]**2 * rhoDM_tmp[j] for j in range(0, len(rDM_tmp))] # [Msun/kpc]
	MrDM_int_interp = interpolate.InterpolatedUnivariateSpline(x=rDM_tmp, y=MrDM_int, k=3)
	MrDM_tmp = [MrDM_int_interp.integral(0, el) for el in rDM_tmp] # [Msun/kpc]
	MrDM_interp = interpolate.InterpolatedUnivariateSpline(x=rDM_tmp, y=MrDM_tmp, k=3)

	# since G is in code units [kpc * km^2 / 1e10 Msun / s^2], we have to multiply the total mass by 1.0/(1.0e10)
	def Mtot(r, YoYSPS=YoYSPS0) :
		mtot = Mstars(r, YoYSPS) + MrDM_interp(r)
		return mtot / (1.0e10) # [1e10 Msun]


	# better to use rDM_tmp instead of r_list!
	def SigmasVelDisp2losR(R, beta_gen=betagen, YoYSPS=YoYSPS0) :
		SigmasVelDisp2los_int = [2.0 * G * Fbeta(rDM_tmp[j], R, beta_gen) * Mtot(rDM_tmp[j], YoYSPS) * rhos(rDM_tmp[j]) / (rDM_tmp[j])**(2.0 - 2.0 * beta_gen) for j in range(0, len(rDM_tmp))]
		SigmasVelDisp2los_int_interp = interpolate.InterpolatedUnivariateSpline(x=rDM_tmp, y=SigmasVelDisp2los_int, k=3) 
		SigmasVelDisp2los = SigmasVelDisp2los_int_interp.integral(R, np.inf) # [Lsun * km^2 / kpc^2 / s^2]
		return SigmasVelDisp2los


	# interpolate previous result (beta_gen and YoYSPS are fixed from here on!) #
	SigmasVelDisp2losR_list = [SigmasVelDisp2losR(el, betagen, YoYSPS0) for el in r_list]
	SigmasVelDisp2losR_interp = interpolate.InterpolatedUnivariateSpline(x=r_list, y=SigmasVelDisp2losR_list, k=3)


	#Sigmas_list = [Sigmas(el) for el in r_SIDM]
	#Mtot_list = [Mtot_SIDM(el, YoYSPS0) for el in r_SIDM]
	#VelDisplos_list = [np.sqrt(SigmasVelDisp2losR_list[j] / Sigmas_list[j]) for j in range(0, len(r_SIDM))]
	#print "\nr_SIDM = ", r_SIDM
	##print "\nMtot_list = ", Mtot_list
	##print "\nSigmas_list = ", Sigmas_list
	##print "\nSigmasVelDisp2losR_list = ", SigmasVelDisp2losR_list
	#print "\nVelDisplos_list = ", VelDisplos_list
	##sys.exit()


	def SigmasR_tilde(R, sigma_PSF=sigmaPSF) :
		i0pt = [R * r_list[j] / sigma_PSF**2 for j in range(0, len(r_list))]
		i0_list = [sc.i0(el) * np.exp(-el) for el in i0pt]
		Sigmas_tilde_int = [r_list[j] * Sigmas(r_list[j]) * i0_list[j] * np.exp(- (R - r_list[j])**2 / (2.0 * sigma_PSF**2) ) for j in range(0, len(r_list))]
		Sigmas_tilde_int_interp = interpolate.InterpolatedUnivariateSpline(x=r_list, y=Sigmas_tilde_int, k=3) 
		Sigmas_tilde = Sigmas_tilde_int_interp.integral(0.0, np.inf) # [Lsun * km^2 / kpc^2 / s^2]
		return Sigmas_tilde


	# interpolate previous result (sigma_PSF is fixed from here on!) #
	SigmasR_tilde_list = [SigmasR_tilde(el, sigmaPSF) for el in Rlist]
	SigmasR_tilde_interp = interpolate.InterpolatedUnivariateSpline(x=Rlist, y=SigmasR_tilde_list, k=3)


	#print "SigmasR_tilde_list = ", SigmasR_tilde_list
	#i0pt = [Rlist[-1] * r_SIDM[j] / sigmaPSF**2 for j in range(0, len(r_SIDM))]
	#print "\ni0pt = ", i0pt
	#i0_list = [sc.i0(el) * np.exp(-el) for el in i0pt]
	#print "i0_list = ", i0_list
	#sys.exit()


	def SigmasVelDisp2losR_tilde(R, sigma_PSF=sigmaPSF) :
		i0pt = [R * r_list[j] / sigma_PSF**2 for j in range(0, len(r_list))]
		i0_list = [sc.i0(el) * np.exp(-el) for el in i0pt]
		SigmasVelDisp2los_tilde_int = [r_list[j] * SigmasVelDisp2losR_interp(r_list[j]) * i0_list[j] * np.exp(- (R - r_list[j])**2 / (2.0 * sigma_PSF**2) ) for j in range(0, len(r_list))]
		SigmasVelDisp2los_tilde_int_interp = interpolate.InterpolatedUnivariateSpline(x=r_list, y=SigmasVelDisp2los_tilde_int, k=3) 
		SigmasVelDisp2los_tilde = SigmasVelDisp2los_tilde_int_interp.integral(0.0, np.inf) # [Lsun * km^2 / kpc^2 / s^2]
		return SigmasVelDisp2los_tilde


	# interpolate previous result (beta_gen, YoYSPS and sigma_PSF are fixed from here on!) #
	SigmasVelDisp2losR_tilde_list = [SigmasVelDisp2losR_tilde(el, sigmaPSF) for el in Rlist]
	SigmasVelDisp2losR_tilde_interp = interpolate.InterpolatedUnivariateSpline(x=Rlist, y=SigmasVelDisp2losR_tilde_list, k=3)


	## now compute final integrals ##
	def veldispLOS2_num_y(x, w_slit=wslit) :
		Rlist = [np.sqrt(x**2 + el**2) for el in ylist]
		num_y_int = [SigmasVelDisp2losR_tilde_interp(el) for el in Rlist]
		num_y_int_interp = interpolate.InterpolatedUnivariateSpline(x=ylist, y=num_y_int, k=3)
		num_y = num_y_int_interp.integral(0, wslit/2.0) 
		return 2.0 * num_y


	def veldispLOS2_num(R_min, R_max, w_slit=wslit) :
		num_int = [veldispLOS2_num_y(el, w_slit) for el in xlist]
		num_int_interp = interpolate.InterpolatedUnivariateSpline(x=xlist, y=num_int, k=3)
		num = num_int_interp.integral(R_min, R_max) 
		return num


	def veldispLOS2_den_y(x, w_slit=wslit) :
		Rlist = [np.sqrt(x**2 + el**2) for el in ylist]
		den_y_int = [SigmasR_tilde_interp(el) for el in Rlist]
		den_y_int_interp = interpolate.InterpolatedUnivariateSpline(x=ylist, y=den_y_int, k=3)
		den_y = den_y_int_interp.integral(0, wslit/2.0) 
		return 2.0 * den_y


	def veldispLOS2_den(R_min, R_max, w_slit=wslit) :
		den_int = [veldispLOS2_den_y(el, w_slit) for el in xlist]
		den_int_interp = interpolate.InterpolatedUnivariateSpline(x=xlist, y=den_int, k=3)
		den = den_int_interp.integral(R_min, R_max) 
		return den


	def veldispLOS_fin(R_min, R_max, w_slit=wslit) :
		num_i = veldispLOS2_num(R_min, R_max, w_slit)
		den_i = veldispLOS2_den(R_min, R_max, w_slit)
		return np.sqrt(num_i / den_i)


	return [veldispLOS_fin(R_min=Rmin[j], R_max=Rmax[j], w_slit=wslit) for j in range(0, len(Rmin))] 



# comparison with original data for SIDM #
veldispLOS_SIDM_cmp = veldispLOS(logr=logr_SIDM_c, logrho=logrho_SIDM_c, Rmin=dcr_data_exp_l, Rmax=dcr_data_exp_r)
print "\ncr_data_exp = ", cr_data_exp
print "veldispLOS_SIDM_cmp = ", veldispLOS_SIDM_cmp



## baryon density ##
rb_list = np.logspace(start=np.log10(min(r_SIDM_bar_c)), stop=np.log10(max(r_SIDM_bar_c)), num=1000)
rhob_dPIE = [rhob_s(r=el, YoYSPS=YoYSPS0, Y_SPS=YSPS) for el in rb_list]

drho_SIDM_bar_1p_c = [(rho_SIDM_bar_c[j] + drho_SIDM_bar_c[j]) for j in range(0, len(r_SIDM_bar_c))]
drho_SIDM_bar_1m_c = [(rho_SIDM_bar_c[j] - drho_SIDM_bar_c[j]) for j in range(0, len(r_SIDM_bar_c))]




# Density analysis #
# VECTOR #
mV_text = r'$m_V = 27$ MeV'
infile_1 = "test3/hernquist/vector/A2537_0015_27/out/snp_013"



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
PPosDM_1, PVelDM_1, _, _ = readGadget1.readSnapshot(filename=infile_1, ptype='dm', strname='full', full=True, mass=False)


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
numbins1 = 30
mp *= 1.0e10 # Msun

r_min_1 = np.around(a=min(pos_r_1), decimals=1)
r_max_1 = np.around(a=max(pos_r_1), decimals=0) + 1.0
print "\nr_min = ", r_min_1, " kpc"
print "r_max = ", r_max_1, " kpc"


r_max = 100.0 * a
if r_min_1 == 0.0 :
	r_min = 0.05
else :
	r_min = 1.0 # kpc


logbins = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=numbins1) # left edges

Npart_bins_1 = [0 for j in range(0, len(logbins)-1)]
vel_bins_1 = [[] for j in range(0, len(logbins)-1)]
pos_bins_1 = [[] for j in range(0, len(logbins)-1)]
for i in range(0, len(pos_r_1)) :	
	for j in range(0, len(logbins)-1) :
		if pos_r_1[i] >= logbins[j] and pos_r_1[i] < logbins[j+1] :
			Npart_bins_1[j] += 1
			vel_bins_1[j].append(vel_r_1[i])
			pos_bins_1[j].append(pos_r_1[i])
			break

dNpart_bins_1 = [np.sqrt(el) for el in Npart_bins_1]


logmid_bins = []
dlogmid_bins = []
vel_mean_bins_1 = []
dvel_mean_bins_1 = []
rho_bins_1 = []
drho_bins_1 = []
for j in range(0, len(logbins)-1) :
	vel_sum_j = sum(vel_bins_1[j]) # len(vel_bins[j]) == Npart_bins[j]
	hwidth_j = 0.5 * (logbins[j+1] - logbins[j])
	logmid_bins.append(logbins[j] + hwidth_j)
	dlogmid_bins.append(hwidth_j)
	if (Npart_bins_1[j] != 0) :
		vel_mean_bins_1.append(vel_sum_j/Npart_bins_1[j])
		dvel_mean_bins_1.append(vel_sum_j/(Npart_bins_1[j]**1.5))
		mass_j = Npart_bins_1[j] * mp
		dmass_j = np.sqrt(Npart_bins_1[j]) * mp # Poisson uncertainty of N counts
		vol_j = 4.0 * np.pi / 3.0 * (logbins[j+1]**3 - logbins[j]**3)
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


# NFW
#rhoNFW_th = [rhos(s=mid_bins[j]/rv1, c=c1, Mv=Mv1*1.0e10, rv=rv1) for j in range(0, len(bins)-1)]
rhoNFW_th = [rhoNFW(r=logmid_bins[j], rhoss=rhos1*1.0e10, rs=rs1) for j in range(0, len(logbins)-1)]
veldispNFW_th = [veldisps(s=logmid_bins[j]/(c1 * rs1), c=c1, Vv=Vv1) for j in range(0, len(logbins)-1)]

# Hernquist
rhoH_th = [rhoH(r=logmid_bins[j], M=M*1.0e10, a=a) for j in range(0, len(logbins)-1)]
veldispH_th = [veldispH(r=logmid_bins[j], M=M, a=a) for j in range(0, len(logbins)-1)] # G*M has right units #/np.sqrt(3.0) would work!


drho_bins_1_1p = [(rho_bins_1[j] + drho_bins_1[j]) for j in range(0, len(logbins)-1)]
drho_bins_1_1m = [(rho_bins_1[j] - drho_bins_1[j]) for j in range(0, len(logbins)-1)]
dvel_std_bins_1_1p = [(vel_std_bins_1[j] + dvel_std_bins_1[j]) for j in range(0, len(logbins)-1)]
dvel_std_bins_1_1m = [(vel_std_bins_1[j] - dvel_std_bins_1[j]) for j in range(0, len(logbins)-1)]


llogmid_bins = [np.log10(el) for el in logmid_bins]
logrhoH_th = [np.log10(el * convfac) for el in rhoH_th]
logrhoNFW_th = [np.log10(el * convfac) for el in rhoNFW_th]

logrho_bins_1 = [np.log10(el * convfac) if el > 0 else -2.0 for el in rho_bins_1]
logdrho_bins_1_1p = [np.log10(drho_bins_1_1p[j] * convfac) if drho_bins_1_1p[j] != 0 else 0 for j in range(0, len(logbins)-1)]
logdrho_bins_1_1m = [np.log10(drho_bins_1_1m[j] * convfac) if drho_bins_1_1m[j] != 0 else 0 for j in range(0, len(logbins)-1)]





# SIDM #
veldispLOS_SIDM = veldispLOS(logr=logr_SIDM_c, logrho=logrho_SIDM_c, Rmin=binsL, Rmax=binsR)

# VECTOR #
veldispLOS_bins_1 = veldispLOS(logr=llogmid_bins, logrho=logrho_bins_1, Rmin=binsL, Rmax=binsR)
veldispLOS_bins_1_1p = veldispLOS(logr=llogmid_bins, logrho=logdrho_bins_1_1p, Rmin=binsL, Rmax=binsR)
veldispLOS_bins_1_1m = veldispLOS(logr=llogmid_bins, logrho=logdrho_bins_1_1m, Rmin=binsL, Rmax=binsR)


print "\nMsun = ", Msun, " GeV"
print "kpc = ", kpc, " cm"




##-- Plots --##
fig1 = plt.figure(num='rhob_SIDM_cmp', figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax1.set_ylabel(r'$\rho_{b} (r)$ [M$_{\odot}$/kpc$^{3}$]', fontsize=20)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1.0, 14.0)
ax1.set_ylim(8.0e6, 2.5e9)
ax1.plot(r_SIDM_bar_c, rho_SIDM_bar_c, color ='orange', linestyle = '-', mec='black', lw=3.0, label=r'SIDM fit 1508.03339')
ax1.fill_between(r_SIDM_bar_c, drho_SIDM_bar_1m_c, drho_SIDM_bar_1p_c, color ='orange', alpha=0.3)
ax1.plot(rb_list, rhob_dPIE, color ='red', linestyle = '-', mec='black', lw=2.0, label=r'Best fit dPIE used in 1209.1391')
ax1.legend(loc='upper right', prop={'size': 18})
ax1.grid(False)
ax1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
fig1.tight_layout()
fig1.show()
#fig1.savefig('test3/figs/rhob_SIDM_vs_r_A2537_cmp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



# VECTOR #
text1 = r'A 2537 -- Model 1'
text2 = r'$m_X = 100$ MeV' + '\n' + r'$\alpha\prime = 0.015$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'
text3 = r'$r_s = {0:0.1f}$'.format(rs1) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$V_v = {0:0.0f}$'.format(Vv1) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1) + r' kpc'


r_max = float(r_max / 10.0)

##-- Plots --##
fig3 = plt.figure(num='stdv_vs_r', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax3.set_ylabel(r'$\sigma_{v_r} (r)$ [km/s]', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(4.0, 5.0e3)
ax3.set_ylim(5.0e1, 2.0e3)
ax3.plot(logmid_bins, veldispH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax3.plot(logmid_bins, veldispNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax3.plot(logmid_bins, vel_std_bins_1, color='darkviolet', linestyle='-', lw=2.0, label=mV_text)
ax3.fill_between(logmid_bins, dvel_std_bins_1_1m, dvel_std_bins_1_1p, color ='darkviolet', alpha=0.3)
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
ax4.set_xlim(5.0, 1.0e3)
ax4.set_ylim(1.0e5, 5.0e8)
ax4.plot(logmid_bins, rhoH_th, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax4.plot(logmid_bins, rhoNFW_th, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax4.plot(logmid_bins, rho_bins_1, color='darkviolet', linestyle='-', lw=2.0, label=mV_text)
ax4.fill_between(logmid_bins, drho_bins_1_1m, drho_bins_1_1p, color ='darkviolet', alpha=0.3)
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.3, y=1.0e6, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
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
ax4_1.set_xlim(0.6, 2.6)
ax4_1.set_ylim(-1.9, 1.4)
ax4_1.plot(llogmid_bins, logrhoH_th, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax4_1.plot(llogmid_bins, logrhoNFW_th, color ='black', linestyle =':', lw=2.0, label=r'NFW')
ax4_1.plot(logr_SIDM_c, logrho_SIDM_c, color ='orange', linestyle ='-', mec='black', lw=3.0, label=r'SIDM fit 1508.03339')

ax4_1.plot(llogmid_bins, logrho_bins_1, color='darkviolet', linestyle='-', lw=2.0, label=mV_text)
ax4_1.fill_between(llogmid_bins, logdrho_bins_1_1m, logdrho_bins_1_1p, color ='darkviolet', alpha=0.3)
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
#ax4_1.legend(loc='upper right', prop={'size': 16})
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





# velocity dispersion #
fig5 = plt.figure(num='veldispLOS_vector', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$\sigma_{\rm LOS}^{\star} (r)$ [km/s]', fontsize=20)
ax5.set_xscale('log')
ax5.set_yscale('linear')
ax5.set_xlim(0.5, 17.5)
ax5.set_ylim(200.0, 575.0)

xerr_exp = np.array([[cr_data_exp[0]-dcr_data_exp_l[0], dcr_data_exp_r[0]-cr_data_exp[0]], [cr_data_exp[1]-dcr_data_exp_l[1], dcr_data_exp_r[1]-cr_data_exp[1]], [cr_data_exp[2]-dcr_data_exp_l[2], dcr_data_exp_r[2]-cr_data_exp[2]], [cr_data_exp[3]-dcr_data_exp_l[3], dcr_data_exp_r[3]-cr_data_exp[3]], [cr_data_exp[4]-dcr_data_exp_l[4], dcr_data_exp_r[4]-cr_data_exp[4]]]).T
data5=make_error_boxes(ax5, cr_data_exp, proj_stdv_data_exp, xerror=xerr_exp, yerror=dproj_stdv_data_exp, facecolor='grey', edgecolor='grey', alpha=0.5, label='Data')
#ax5.errorbar(cr_data_exp, proj_stdv_data_exp, xerr=xerr_exp, yerr=dproj_stdv_data_exp, c='black', marker='o', mec='black', alpha=1.0, linestyle='None')

ax5.plot(r_SIDM_fit_cpm, vdisp_SIDM_fit_c, color ='orange', linestyle = '-', mec='black', lw=3.0, label=r'SIDM fit 1508.03339')
#ax5.fill_between(r_SIDM_fit_cpm, vdisp_SIDM_fit_m, vdisp_SIDM_fit_p, color ='orange', alpha=0.3)
##ax5.plot(mid_bins, veldispLOS_SIDM, color ='red', linestyle = '-', mec='black', lw=3.0, label=r'Best fit dPIE used in 1209.1391')
##ax5.legend(loc='lower right', prop={'size': 18})

ax5.plot(mid_bins, veldispLOS_bins_1, color ='darkviolet', linestyle ='-.', lw=2.0, label=mV_text)
ax5.fill_between(mid_bins, veldispLOS_bins_1_1m, veldispLOS_bins_1_1p, color ='darkviolet', alpha=0.3)
ax5.legend(loc='upper left', prop={'size': 18})

#ob5 = offsetbox.AnchoredText(s=text2_sc, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5 = offsetbox.AnchoredText(s=text2, loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
#ob5_1 = offsetbox.AnchoredText(s=text1_sc, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1 = offsetbox.AnchoredText(s=text1, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)

ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50.0))
ax5.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10.0))
fig5.tight_layout()
fig5.show()




raw_input()
