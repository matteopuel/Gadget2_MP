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




#------------#
### DDO154 ###
#------------#

# Hernquist and simulation parameters in code units!
#------# (real simulation: DDO 154) #-------#
Mv1_g = 2.3 # * 1e10 Msun (total mass -> dSph galaxy)
rv1_g = R200(Mv1_g) # kpc
rhos1_g = 1.5e-3 # * 1e10 Msun/kpc^3
rs1_g = 3.4 # kpc
Vv1_g = 49.0 # km/s
c1_g = 12.2

eps_g = 0.3 # kpc

# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
## using inferred M,a from rhos1 and rs1
infile_g = "test3/hernquist/hernquist_v1_S1"
M_g = 1.226850076349031
a_g = 6.187578545092555


## VECTOR ('_v') for GALAXY ('g') ##
#----------------------------------#
# sigma_T
vgname = 'test3/hernquist/data/DDO154_vector_sim.dat'

logmid_bins_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
logrhoH_th_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
logrhoNFW_th_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
logrho_bins_1_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 3)
logdrho_bins_1_1p_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 4)
logdrho_bins_1_1m_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 5)
logrho_bins_2_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 6)
logdrho_bins_2_1p_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 7)
logdrho_bins_2_1m_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 8)
logrho_bins_3_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 9)
logdrho_bins_3_1p_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 10)
logdrho_bins_3_1m_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 11)


# NFW (Jim - vector) #
logr_SIDM_g = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 0)
logrho_SIDM_g = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 1)

logr_26MeV_vg = np.loadtxt(fname='../../DDO154_26MeV.dat', delimiter=' ', usecols = 0)
logrho_26MeV_vg = np.loadtxt(fname='../../DDO154_26MeV.dat', delimiter=' ', usecols = 1)
logr_34MeV_vg = np.loadtxt(fname='../../DDO154_34MeV.dat', delimiter=' ', usecols = 0)
logrho_34MeV_vg = np.loadtxt(fname='../../DDO154_34MeV.dat', delimiter=' ', usecols = 1)
logr_17MeV_vg = np.loadtxt(fname='../../DDO154_17MeV.dat', delimiter=' ', usecols = 0)
logrho_17MeV_vg = np.loadtxt(fname='../../DDO154_17MeV.dat', delimiter=' ', usecols = 1)
# Hernquist (Jim - vector) #
logr_26MeV_H_vg = np.loadtxt(fname='../../rho-dwarf-vector-hern.dat', delimiter='\t\t\t', usecols = 0)
logrho_26MeV_H_vg = np.loadtxt(fname='../../rho-dwarf-vector-hern.dat', delimiter='\t\t\t', usecols = 2)



#text1_vg = r'$r_s = {0:0.1f}$'.format(rs1_g) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_g*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_g) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_g) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_g*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1_g) + r' kpc'
text1_vg = r'DDO 154 -- Model 1'
text2_vg = r'$m_X = 100$ MeV' + '\n' + r'$\alpha^{\prime} = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig1 = plt.figure(num='logrho_vs_r_DDO154_vector', figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax1.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax1.set_xscale('linear')
ax1.set_yscale('linear')
ax1.set_xlim(-1.0, 0.9)
ax1.set_ylim(-1.9, 1.4)
ax1.plot(logmid_bins_vg, logrhoNFW_th_vg, color ='black', linestyle ='-', lw=2.0, label=r'original NFW')
ax1.plot(logmid_bins_vg, logrhoH_th_vg, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax1.plot(logr_SIDM_g, logrho_SIDM_g, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')

ax1.plot(logmid_bins_vg, logrho_bins_1_vg, color='violet', linestyle='-', lw=2.0, label=r'$m_V = 26$ MeV (MP, H)')
ax1.fill_between(logmid_bins_vg, logdrho_bins_1_1m_vg, logdrho_bins_1_1p_vg, color ='violet', alpha=0.3)
ax1.plot(logr_26MeV_vg, logrho_26MeV_vg, color ='darkviolet', linestyle ='-.', lw=2.0, label=r'$m_V = 26$ MeV (JC, NFW)')
#ax1.plot(logr_26MeV_H_vg, logrho_26MeV_H_vg, color ='steelblue', linestyle =':', lw=2.0, label=r'$m_V = 26$ MeV (JC, H)')
ax1.plot(logmid_bins_vg, logrho_bins_2_vg, color='cyan', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV (MP, H)')
ax1.fill_between(logmid_bins_vg, logdrho_bins_2_1m_vg, logdrho_bins_2_1p_vg, color ='cyan', alpha=0.3)
ax1.plot(logr_34MeV_vg, logrho_34MeV_vg, color ='blue', linestyle ='-.', lw=2.0, label=r'$m_V = 34$ MeV (JC, NFW)')
ax1.plot(logmid_bins_vg, logrho_bins_3_vg, color='lime', linestyle='-', lw=2.0, label=r'$m_V = 17$ MeV (MP, H)')
ax1.fill_between(logmid_bins_vg, logdrho_bins_3_1m_vg, logdrho_bins_3_1p_vg, color ='lime', alpha=0.3)
ax1.plot(logr_17MeV_vg, logrho_17MeV_vg, color ='green', linestyle ='-.', lw=2.0, label=r'$m_V = 17$ MeV (JC, NFW)')
ax1.legend([("red","--", " ", " "), ("black","-", " ", " "), ("orange","-", " ", " "), ("cyan","-", "blue","-."), ("violet","-", "darkviolet","-."), ("lime","-", "green","-.")], [r'Hernquist', r'original NFW', r'SIDM fit 1611.02716', r'$m_V = 34$ MeV', r'$m_V = 26$ MeV', r'$m_V = 17$ MeV'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})

ax1.axvline(np.log10(eps_g), color='gray', linestyle = '--', lw=2.0)
ax1.text(x=np.log10(eps_g) + 0.03, y=1.1, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax1.grid(False)
ax1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
#ob1 = offsetbox.AnchoredText(s=text2_vg, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob1 = offsetbox.AnchoredText(s=text2_vg, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax1.add_artist(ob1)
#ob1_1 = offsetbox.AnchoredText(s=text1_vg, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob1_1 = offsetbox.AnchoredText(s=text1_vg, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob1_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax1.add_artist(ob1_1)
fig1.tight_layout()
fig1.show()
#fig1.savefig('test3/figs/logrho_vs_r_DDO154_vector.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



## SCALAR ('_s') for GALAXY ('g') ##
#----------------------------------#
## sigma_T
#sgname = 'test3/hernquist/data/DDO154_scalar_sim.dat'
sgname = 'test3/hernquist/data/DDO154_scalar_sim_NEW.dat'

logmid_bins_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
logrhoH_th_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
logrhoNFW_th_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
logrho_bins_1_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 3)
logdrho_bins_1_1p_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 4)
logdrho_bins_1_1m_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 5)
logrho_bins_2_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 6)
logdrho_bins_2_1p_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 7)
logdrho_bins_2_1m_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 8)
logrho_bins_3_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 9)
logdrho_bins_3_1p_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 10)
logdrho_bins_3_1m_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 11)


# NFW (Jim - scalar) #
logr_001_sg = np.loadtxt(fname='../../DDO154s_001.dat', delimiter=' ', usecols = 0)
logrho_001_sg = np.loadtxt(fname='../../DDO154s_001.dat', delimiter=' ', usecols = 1)
logr_0015_sg = np.loadtxt(fname='../../DDO154s_0015.dat', delimiter=' ', usecols = 0)
logrho_0015_sg = np.loadtxt(fname='../../DDO154s_0015.dat', delimiter=' ', usecols = 1)
logr_002_sg = np.loadtxt(fname='../../DDO154s_002.dat', delimiter=' ', usecols = 0)
logrho_002_sg = np.loadtxt(fname='../../DDO154s_002.dat', delimiter=' ', usecols = 1)



#text1_sg = r'$r_s = {0:0.1f}$'.format(rs1_g) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_g*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_g) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_g) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_g*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1_g) + r' kpc'
text1_sg = r'DDO 154 -- Model 2'
text2_sg = r'$m_X = 100$ MeV' + '\n' + r'$m_{\phi} \lesssim 2\,m_{\chi}$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig2 = plt.figure(num='logrho_vs_r_DDO154_scalar', figsize=(10, 7), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax2.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax2.set_xscale('linear')
ax2.set_yscale('linear')
ax2.set_xlim(-1.0, 0.9)
ax2.set_ylim(-1.9, 1.4)
ax2.plot(logmid_bins_sg, logrhoNFW_th_sg, color ='black', linestyle ='-', lw=2.0, label=r'original NFW')
ax2.plot(logmid_bins_sg, logrhoH_th_sg, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax2.plot(logr_SIDM_g, logrho_SIDM_g, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')

ax2.plot(logmid_bins_sg, logrho_bins_1_sg, color='cyan', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.01$ (MP, H)')
ax2.fill_between(logmid_bins_sg, logdrho_bins_1_1m_sg, logdrho_bins_1_1p_sg, color ='cyan', alpha=0.3)
ax2.plot(logr_001_sg, logrho_001_sg, color ='blue', linestyle ='-.', lw=2.0, label=r'$\alpha^{\prime} = 0.01$ (JC, NFW)')
ax2.plot(logmid_bins_sg, logrho_bins_2_sg, color='violet', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.015$ (MP, H)')
ax2.fill_between(logmid_bins_sg, logdrho_bins_2_1m_sg, logdrho_bins_2_1p_sg, color ='violet', alpha=0.3)
ax2.plot(logr_0015_sg, logrho_0015_sg, color ='darkviolet', linestyle ='-.', lw=2.0, label=r'$\alpha^{\prime} = 0.015$ (JC, NFW)')
ax2.plot(logmid_bins_sg, logrho_bins_3_sg, color='lime', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.02$ (MP, H)')
ax2.fill_between(logmid_bins_sg, logdrho_bins_3_1m_sg, logdrho_bins_3_1p_sg, color ='lime', alpha=0.3)
ax2.plot(logr_002_sg, logrho_002_sg, color ='green', linestyle ='-.', lw=2.0, label=r'$\alpha^{\prime} = 0.02$ (JC, NFW)')
ax2.legend([("red","--", " ", " "), ("black","-", " ", " "), ("orange","-", " ", " "), ("cyan","-", "blue","-."), ("violet","-", "darkviolet","-."), ("lime","-", "green","-.")], [r'Hernquist', r'original NFW', r'SIDM fit 1611.02716', r'$\alpha^{\prime} = 0.01$', r'$\alpha^{\prime} = 0.015$', r'$\alpha^{\prime} = 0.02$'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})

ax2.axvline(np.log10(eps_g), color='gray', linestyle = '--', lw=2.0)
ax2.text(x=np.log10(eps_g) + 0.03, y=1.1, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax2.grid(False)
ax2.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax2.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax2.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax2.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
#ob2 = offsetbox.AnchoredText(s=text2_sg, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob2 = offsetbox.AnchoredText(s=text2_sg, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob2.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax2.add_artist(ob2)
#ob2_1 = offsetbox.AnchoredText(s=text1_sg, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob2_1 = offsetbox.AnchoredText(s=text1_sg, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob2_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax2.add_artist(ob2_1)
fig2.tight_layout()
fig2.show()
#fig2.savefig('test3/figs/logrho_vs_r_DDO154_scalar.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



#-----------#
### A2537 ###
#-----------#

# Hernquist and simulation parameters in code units!
##---------# (A 2537 galaxy cluster) #---------##
rv1_c = 2050.0 # kpc
Mv1_c = M200(rv1_c) # * 1e10 Msun (total mass -> galaxy cluster)
rhos1_c = 1.3e-4 # * 1e10 Msun/kpc^3
rs1_c = 442.0 # kpc
Vv1_c = 1660.0 # km/s
c1_c = 4.63

eps_c = 9.3 # kpc

# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
## using inferred M,a from rhos1 and rs1
infile_c = "test3/hernquist/hernquist_v1_S1_cluster"
M_c = 127766.92971544608 # * 1e10 Msun
a_c = 594.8897476275504 # kpc


## VECTOR ('_v') for CLUSTER ('c') ##
#-----------------------------------#
# sigma_T
vcname = 'test3/hernquist/data/A2537_vector_sim.dat'

logmid_bins_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
logrhoH_th_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
logrhoNFW_th_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
logrho_bins_1_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 3)
logdrho_bins_1_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 4)
logdrho_bins_1_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 5)
logrho_bins_2_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 6)
logdrho_bins_2_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 7)
logdrho_bins_2_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 8)
logrho_bins_3_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 9)
logdrho_bins_3_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 10)
logdrho_bins_3_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 11)
logrho_bins_4_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 12)
logdrho_bins_4_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 13)
logdrho_bins_4_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 14)


# NFW (Jim - vector) #
logr_SIDM_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 0)
logrho_SIDM_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 1)
logdrho_SIDM_1p_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 2)
logdrho_SIDM_1m_c = np.loadtxt(fname='../../A2537_rho_SIDM_dm.dat', delimiter='\t', usecols = 3)

logr_68MeV_vc = np.loadtxt(fname='../../A2537_68MeV.dat', delimiter=' ', usecols = 0)
logrho_68MeV_vc = np.loadtxt(fname='../../A2537_68MeV.dat', delimiter=' ', usecols = 1)
logr_51MeV_vc = np.loadtxt(fname='../../A2537_51MeV.dat', delimiter=' ', usecols = 0)
logrho_51MeV_vc = np.loadtxt(fname='../../A2537_51MeV.dat', delimiter=' ', usecols = 1)
logr_43MeV_vc = np.loadtxt(fname='../../A2537_43MeV.dat', delimiter=' ', usecols = 0)
logrho_43MeV_vc = np.loadtxt(fname='../../A2537_43MeV.dat', delimiter=' ', usecols = 1)
logr_34MeV_vc = np.loadtxt(fname='../../A2537_34MeV.dat', delimiter=' ', usecols = 0)
logrho_34MeV_vc = np.loadtxt(fname='../../A2537_34MeV.dat', delimiter=' ', usecols = 1)



#text1_vc = r'$r_s = {0:0.0f}$'.format(rs1_c) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_c*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_c) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_c) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_c*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.0f}$'.format(rv1_c) + r' kpc'
text1_vc = r'A 2537 -- Model 1'
text2_vc = r'$m_X = 100$ MeV' + '\n' + r'$\alpha^{\prime} = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig3 = plt.figure(num='logrho_vs_r_A2537_vector', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax3.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax3.set_xscale('linear')
ax3.set_yscale('linear')
#ax3.set_xlim(0.42, 2.6)
ax3.set_xlim(0.58, 2.6)
ax3.set_ylim(-1.9, 1.4)
ax3.plot(logmid_bins_vc, logrhoNFW_th_vc, color ='black', linestyle ='-', lw=2.0, label=r'original NFW')
ax3.plot(logmid_bins_vc, logrhoH_th_vc, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax3.plot(logr_SIDM_c, logrho_SIDM_c, color ='orange', linestyle ='-', mec='black', lw=3.0, label=r'SIDM fit 1508.03339')
#ax3.fill_between(logr_SIDM_c, logdrho_SIDM_1m_c, logdrho_SIDM_1p_c, color ='orange', alpha=0.3)

ax3.plot(logmid_bins_vc, logrho_bins_1_vc, color='cyan', linestyle='-', lw=2.0, label=r'$m_V = 68$ MeV (MP, H)')
ax3.fill_between(logmid_bins_vc, logdrho_bins_1_1m_vc, logdrho_bins_1_1p_vc, color ='cyan', alpha=0.3)
ax3.plot(logr_68MeV_vc, logrho_68MeV_vc, color ='blue', linestyle ='-.', lw=2.0, label=r'$m_V = 68$ MeV (JC, NFW)')
ax3.plot(logmid_bins_vc, logrho_bins_2_vc, color='violet', linestyle='-', lw=2.0, label=r'$m_V = 51$ MeV (MP, H)')
ax3.fill_between(logmid_bins_vc, logdrho_bins_2_1m_vc, logdrho_bins_2_1p_vc, color ='violet', alpha=0.3)
ax3.plot(logr_51MeV_vc, logrho_51MeV_vc, color ='darkviolet', linestyle ='-.', lw=2.0, label=r'$m_V = 51$ MeV (JC, NFW)')
ax3.plot(logmid_bins_vc, logrho_bins_3_vc, color='lime', linestyle='-', lw=2.0, label=r'$m_V = 43$ MeV (MP, H)')
ax3.fill_between(logmid_bins_vc, logdrho_bins_3_1m_vc, logdrho_bins_3_1p_vc, color ='lime', alpha=0.3)
ax3.plot(logr_43MeV_vc, logrho_43MeV_vc, color ='green', linestyle ='-.', lw=2.0, label=r'$m_V = 43$ MeV (JC, NFW)')
ax3.plot(logmid_bins_vc, logrho_bins_4_vc, color='chocolate', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV (MP, H)')
ax3.fill_between(logmid_bins_vc, logdrho_bins_4_1m_vc, logdrho_bins_4_1p_vc, color ='chocolate', alpha=0.3)
ax3.plot(logr_34MeV_vc, logrho_34MeV_vc, color ='brown', linestyle ='-.', lw=2.0, label=r'$m_V = 34$ MeV (JC, NFW)')
ax3.legend([("red","--", " ", " "), ("black","-", " ", " "), ("orange","-", " ", " "), ("cyan","-", "blue","-."), ("violet","-", "darkviolet","-."), ("lime","-", "green","-."), ("chocolate","-", "brown","-.")], [r'Hernquist', r'original NFW', r'SIDM fit 1508.03339', r'$m_V = 68$ MeV', r'$m_V = 51$ MeV', r'$m_V = 43$ MeV', r'$m_V = 34$ MeV'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})

ax3.axvline(np.log10(eps_c), color='gray', linestyle = '--', lw=2.0)
ax3.text(x=np.log10(eps_c) + 0.03, y=1.1, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax3.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax3.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax3.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
#ob3 = offsetbox.AnchoredText(s=text2_vc, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3 = offsetbox.AnchoredText(s=text2_vc, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
#ob3_1 = offsetbox.AnchoredText(s=text1_vc, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3_1 = offsetbox.AnchoredText(s=text1_vc, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3_1)
fig3.tight_layout()
fig3.show()
#fig3.savefig('test3/figs/logrho_vs_r_A2537_vector.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)




## SCALAR ('_s') for CLUSTER ('c') ##
#-----------------------------------#
## sigma_T
#scname = 'test3/hernquist/data/A2537_scalar_sim.dat'
scname = 'test3/hernquist/data/A2537_scalar_sim_NEW.dat'

logmid_bins_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
logrhoH_th_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
logrhoNFW_th_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
logrho_bins_1_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 3)
logdrho_bins_1_1p_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 4)
logdrho_bins_1_1m_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 5)
logrho_bins_2_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 6)
logdrho_bins_2_1p_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 7)
logdrho_bins_2_1m_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 8)
logrho_bins_3_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 9)
logdrho_bins_3_1p_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 10)
logdrho_bins_3_1m_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 11)


# NFW (Jim - scalar) #
logr_001_sc = np.loadtxt(fname='../../A2537s_001.dat', delimiter=' ', usecols = 0)
logrho_001_sc = np.loadtxt(fname='../../A2537s_001.dat', delimiter=' ', usecols = 1)
logr_0015_sc = np.loadtxt(fname='../../A2537s_0015.dat', delimiter=' ', usecols = 0)
logrho_0015_sc = np.loadtxt(fname='../../A2537s_0015.dat', delimiter=' ', usecols = 1)
logr_002_sc = np.loadtxt(fname='../../A2537s_002.dat', delimiter=' ', usecols = 0)
logrho_002_sc = np.loadtxt(fname='../../A2537s_002.dat', delimiter=' ', usecols = 1)



#text1_sc = r'$r_s = {0:0.0f}$'.format(rs1_c) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_c*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_c) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_c) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_c*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.0f}$'.format(rv1_c) + r' kpc'
text1_sc = r'A 2537 -- Model 2'
text2_sc = r'$m_X = 100$ MeV' + '\n' + r'$m_{\phi} \lesssim 2\,m_{\chi}$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig4 = plt.figure(num='logrho_vs_r_A2537_scalar', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax4.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax4.set_xscale('linear')
ax4.set_yscale('linear')
ax4.set_xlim(0.58, 2.6)
ax4.set_ylim(-1.9, 1.4)
ax4.plot(logmid_bins_sc, logrhoNFW_th_sc, color ='black', linestyle ='-', lw=2.0, label=r'original NFW')
ax4.plot(logmid_bins_sc, logrhoH_th_sc, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax4.plot(logr_SIDM_c, logrho_SIDM_c, color ='orange', linestyle ='-', mec='black', lw=3.0, label=r'SIDM fit 1508.03339')
#ax4.fill_between(logr_SIDM_c, logdrho_SIDM_1m_c, logdrho_SIDM_1p_c, color ='orange', alpha=0.3)

ax4.plot(logmid_bins_sc, logrho_bins_1_sc, color='cyan', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.01$ (MP, H)')
ax4.fill_between(logmid_bins_sc, logdrho_bins_1_1m_sc, logdrho_bins_1_1p_sc, color ='cyan', alpha=0.3)
ax4.plot(logr_001_sc, logrho_001_sc, color ='blue', linestyle ='-.', lw=2.0, label=r'$\alpha^{\prime} = 0.01$ (JC, NFW)')
ax4.plot(logmid_bins_sc, logrho_bins_2_sc, color='violet', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.015$ (MP, H)')
ax4.fill_between(logmid_bins_sc, logdrho_bins_2_1m_sc, logdrho_bins_2_1p_sc, color ='violet', alpha=0.3)
ax4.plot(logr_0015_sc, logrho_0015_sc, color ='darkviolet', linestyle ='-.', lw=2.0, label=r'$\alpha^{\prime} = 0.015$ (JC, NFW)')
ax4.plot(logmid_bins_sc, logrho_bins_3_sc, color='lime', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.02$ (MP, H)')
ax4.fill_between(logmid_bins_sc, logdrho_bins_3_1m_sc, logdrho_bins_3_1p_sc, color ='lime', alpha=0.3)
ax4.plot(logr_002_sc, logrho_002_sc, color ='green', linestyle ='-.', lw=2.0, label=r'$\alpha^{\prime} = 0.02$ (JC, NFW)')
ax4.legend([("red","--", " ", " "), ("black","-", " ", " "), ("orange","-", " ", " "), ("cyan","-", "blue", "-."), ("violet","-", "darkviolet","-."), ("lime","-", "green","-.")], [r'Hernquist', r'original NFW', r'SIDM fit 1508.03339', r'$\alpha^{\prime} = 0.01$', r'$\alpha^{\prime} = 0.015$', r'$\alpha^{\prime} = 0.02$'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})


ax4.axvline(np.log10(eps_c), color='gray', linestyle = '--', lw=2.0)
ax4.text(x=np.log10(eps_c) + 0.03, y=1.1, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax4.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax4.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
#ob4 = offsetbox.AnchoredText(s=text2_sc, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4 = offsetbox.AnchoredText(s=text2_sc, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
#ob4_1 = offsetbox.AnchoredText(s=text1_sc, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1 = offsetbox.AnchoredText(s=text1_sc, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()
#fig4.savefig('test3/figs/logrho_vs_r_A2537_scalar.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



raw_input()
