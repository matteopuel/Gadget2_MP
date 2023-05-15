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
vgname = 'test3/hernquist/data/DDO154_vector_stdv.dat'

mid_bins_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
veldispH_th_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
veldispNFW_th_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
vel_std_bins_1_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 3)
dvel_std_bins_1_1p_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 4)
dvel_std_bins_1_1m_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 5)
vel_std_bins_2_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 6)
dvel_std_bins_2_1p_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 7)
dvel_std_bins_2_1m_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 8)
vel_std_bins_3_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 9)
dvel_std_bins_3_1p_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 10)
dvel_std_bins_3_1m_vg = np.loadtxt(fname=vgname, delimiter='\t', usecols = 11)


#text1_vg = r'$r_s = {0:0.1f}$'.format(rs1_g) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_g*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_g) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_g) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_g*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1_g) + r' kpc'
text1_vg = r'DDO 154 -- Model 1'
text2_vg = r'$m_X = 100$ MeV' + '\n' + r'$\alpha^{\prime} = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig1 = plt.figure(num='stdv_vs_r_DDO154_vector', figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax1.set_ylabel(r'$\sigma_{r} (r)$ [km/s]', fontsize=20)
ax1.set_xscale('log')
ax1.set_yscale('linear')
ax1.set_xlim(0.1, 3.0e1)
ax1.set_ylim(3.0e0, 4.1e1)
ax1.plot(mid_bins_vg, veldispH_th_vg, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax1.plot(mid_bins_vg, veldispNFW_th_vg, color ='black', linestyle = ':', lw=2.0, label=r'NFW')

ax1.plot(mid_bins_vg, vel_std_bins_2_vg, color='blue', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV')
ax1.fill_between(mid_bins_vg, dvel_std_bins_2_1m_vg, dvel_std_bins_2_1p_vg, color ='blue', alpha=0.3)
ax1.plot(mid_bins_vg, vel_std_bins_1_vg, color='darkviolet', linestyle='-', lw=2.0, label=r'$m_V = 26$ MeV')
ax1.fill_between(mid_bins_vg, dvel_std_bins_1_1m_vg, dvel_std_bins_1_1p_vg, color ='darkviolet', alpha=0.3)
ax1.plot(mid_bins_vg, vel_std_bins_3_vg, color='green', linestyle='-', lw=2.0, label=r'$m_V = 17$ MeV')
ax1.fill_between(mid_bins_vg, dvel_std_bins_3_1m_vg, dvel_std_bins_3_1p_vg, color ='green', alpha=0.3)

ax1.legend(loc='lower right', prop={'size': 18})
ax1.axvline(eps_g, color='gray', linestyle = '--', lw=2.0)
ax1.text(x=eps_g + 0.03, y=15.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax1.grid(False)
ax1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ob1 = offsetbox.AnchoredText(s=text2_vg, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax1.add_artist(ob1)
ob1_1 = offsetbox.AnchoredText(s=text1_vg, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob1_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax1.add_artist(ob1_1)
fig1.tight_layout()
fig1.show()



## SCALAR ('_s') for GALAXY ('g') ##
#----------------------------------#
## sigma_T
sgname = 'test3/hernquist/data/DDO154_scalar_stdv.dat'

mid_bins_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
veldispH_th_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
veldispNFW_th_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
vel_std_bins_1_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 3)
dvel_std_bins_1_1p_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 4)
dvel_std_bins_1_1m_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 5)
vel_std_bins_2_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 6)
dvel_std_bins_2_1p_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 7)
dvel_std_bins_2_1m_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 8)
vel_std_bins_3_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 9)
dvel_std_bins_3_1p_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 10)
dvel_std_bins_3_1m_sg = np.loadtxt(fname=sgname, delimiter='\t', usecols = 11)


#text1_sg = r'$r_s = {0:0.1f}$'.format(rs1_g) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_g*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_g) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_g) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_g*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1_g) + r' kpc'
text1_sg = r'DDO 154 -- Model 2'
text2_sg = r'$m_X = 100$ MeV' + '\n' + r'$m_{\phi} \lesssim 2\,m_{\chi}$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig2 = plt.figure(num='logrho_vs_r_DDO154_scalar', figsize=(10, 7), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel(r'$r$ [kpc]', fontsize=20)
ax2.set_ylabel(r'$\sigma_{r} (r)$ [km/s]', fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('linear')
ax2.set_xlim(0.1, 3.0e1)
ax2.set_ylim(3.0e0, 4.1e1)
ax2.plot(mid_bins_sg, veldispH_th_sg, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax2.plot(mid_bins_sg, veldispNFW_th_sg, color ='black', linestyle = ':', lw=2.0, label=r'NFW')

ax2.plot(mid_bins_sg, vel_std_bins_1_sg, color='blue', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.01$')
ax2.fill_between(mid_bins_sg, dvel_std_bins_1_1m_sg, dvel_std_bins_1_1p_sg, color ='blue', alpha=0.3)
ax2.plot(mid_bins_sg, vel_std_bins_2_sg, color='darkviolet', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.015$')
ax2.fill_between(mid_bins_vg, dvel_std_bins_2_1m_sg, dvel_std_bins_2_1p_sg, color ='darkviolet', alpha=0.3)
ax2.plot(mid_bins_sg, vel_std_bins_3_sg, color='green', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.02$')
ax2.fill_between(mid_bins_sg, dvel_std_bins_3_1m_sg, dvel_std_bins_3_1p_sg, color ='green', alpha=0.3)

ax2.legend(loc='lower right', prop={'size': 18})
ax2.axvline(eps_g, color='gray', linestyle = '--', lw=2.0)
ax2.text(x=eps_g + 0.03, y=15.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax2.grid(False)
ax2.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax2.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax2.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax2.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ob2 = offsetbox.AnchoredText(s=text2_sg, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob2.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax2.add_artist(ob2)
ob2_1 = offsetbox.AnchoredText(s=text1_sg, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob2_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax2.add_artist(ob2_1)
fig2.tight_layout()
fig2.show()



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
vcname = 'test3/hernquist/data/A2537_vector_stdv.dat'

mid_bins_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
veldispH_th_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
veldispNFW_th_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
vel_std_bins_1_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 3)
dvel_std_bins_1_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 4)
dvel_std_bins_1_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 5)
vel_std_bins_2_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 6)
dvel_std_bins_2_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 7)
dvel_std_bins_2_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 8)
vel_std_bins_3_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 9)
dvel_std_bins_3_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 10)
dvel_std_bins_3_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 11)
vel_std_bins_4_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 12)
dvel_std_bins_4_1p_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 13)
dvel_std_bins_4_1m_vc = np.loadtxt(fname=vcname, delimiter='\t', usecols = 14)


#text1_vc = r'$r_s = {0:0.0f}$'.format(rs1_c) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_c*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_c) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_c) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_c*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.0f}$'.format(rv1_c) + r' kpc'
text1_vc = r'A 2537 -- Model 1'
text2_vc = r'$m_X = 100$ MeV' + '\n' + r'$\alpha^{\prime} = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig3 = plt.figure(num='stdv_vs_r_A2537_vector', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$r$ [kpc]', fontsize=20)
ax3.set_ylabel(r'$\sigma_{r} (r)$ [km/s]', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('linear')
ax3.set_xlim(5.0, 1.0e3)
ax3.set_ylim(2.0e2, 1.2e3)
ax3.plot(mid_bins_vc, veldispH_th_vc, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax3.plot(mid_bins_vc, veldispNFW_th_vc, color ='black', linestyle = ':', lw=2.0, label=r'NFW')

ax3.plot(mid_bins_vc, vel_std_bins_1_vc, color='blue', linestyle='-', lw=2.0, label=r'$m_V = 68$ MeV')
ax3.fill_between(mid_bins_vc, dvel_std_bins_1_1m_vc, dvel_std_bins_1_1p_vc, color ='blue', alpha=0.3)
ax3.plot(mid_bins_vc, vel_std_bins_2_vc, color='darkviolet', linestyle='-', lw=2.0, label=r'$m_V = 51$ MeV')
ax3.fill_between(mid_bins_vc, dvel_std_bins_2_1m_vc, dvel_std_bins_2_1p_vc, color ='darkviolet', alpha=0.3)
ax3.plot(mid_bins_vc, vel_std_bins_3_vc, color='green', linestyle='-', lw=2.0, label=r'$m_V = 43$ MeV')
ax3.fill_between(mid_bins_vc, dvel_std_bins_3_1m_vc, dvel_std_bins_3_1p_vc, color ='green', alpha=0.3)
ax3.plot(mid_bins_vc, vel_std_bins_4_vc, color='brown', linestyle='-', lw=2.0, label=r'$m_V = 34$ MeV')
ax3.fill_between(mid_bins_vc, dvel_std_bins_4_1m_vc, dvel_std_bins_4_1p_vc, color ='brown', alpha=0.3)

ax3.legend(loc='lower right', prop={'size': 18})
ax3.axvline(eps_c, color='gray', linestyle = '--', lw=2.0)
ax3.text(x=eps_c + 0.5, y=1.0e3, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ob3 = offsetbox.AnchoredText(s=text2_vc, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
ob3_1 = offsetbox.AnchoredText(s=text1_vc, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3_1)
fig3.tight_layout()
fig3.show()


## SCALAR ('_s') for CLUSTER ('c') ##
#-----------------------------------#
## sigma_T
scname = 'test3/hernquist/data/A2537_scalar_stdv.dat'

mid_bins_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
veldispH_th_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
veldispNFW_th_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 2) # same mid_bins for all of them!
vel_std_bins_1_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 3)
dvel_std_bins_1_1p_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 4)
dvel_std_bins_1_1m_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 5)
vel_std_bins_2_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 6)
dvel_std_bins_2_1p_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 7)
dvel_std_bins_2_1m_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 8)
vel_std_bins_3_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 9)
dvel_std_bins_3_1p_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 10)
dvel_std_bins_3_1m_sc = np.loadtxt(fname=scname, delimiter='\t', usecols = 11)


#text1_sc = r'$r_s = {0:0.0f}$'.format(rs1_c) + r' kpc' + '\n' + r'$\rho_s = {0:s}$'.format(as_si(rhos1_c*1.0e10, 1)) + r' M$_{\odot}$/kpc$^3$' + '\n' + r'$v_v = {0:0.0f}$'.format(Vv1_c) + r' km/s' + '\n' + r'$c_v = {0:0.1f}$'.format(c1_c) + '\n' + r'$M_v = {0:s}$'.format(as_si(Mv1_c*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.0f}$'.format(rv1_c) + r' kpc'
text1_sc = r'A 2537 -- Model 2'
text2_sc = r'$m_X = 100$ MeV' + '\n' + r'$m_{\phi} \lesssim 2\,m_{\chi}$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


fig4 = plt.figure(num='stdv_vs_r_A2537_scalar', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$r$ [kpc]', fontsize=20)
ax4.set_ylabel(r'$\sigma_{r} (r)$ [km/s]', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('linear')
ax4.set_xlim(5.0, 1.0e3)
ax4.set_ylim(2.0e2, 1.2e3)
ax4.plot(mid_bins_sc, veldispH_th_sc, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
ax4.plot(mid_bins_sc, veldispNFW_th_sc, color ='black', linestyle = ':', lw=2.0, label=r'NFW')

ax4.plot(mid_bins_sc, vel_std_bins_1_sc, color='blue', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.01$')
ax4.fill_between(mid_bins_sc, dvel_std_bins_1_1m_sc, dvel_std_bins_1_1p_sc, color ='blue', alpha=0.3)
ax4.plot(mid_bins_sc, vel_std_bins_2_sc, color='darkviolet', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.015$')
ax4.fill_between(mid_bins_sc, dvel_std_bins_2_1m_sc, dvel_std_bins_2_1p_sc, color ='darkviolet', alpha=0.3)
ax4.plot(mid_bins_sc, vel_std_bins_3_sc, color='green', linestyle='-', lw=2.0, label=r'$\alpha^{\prime} = 0.02$')
ax4.fill_between(mid_bins_sc, dvel_std_bins_3_1m_sc, dvel_std_bins_3_1p_sc, color ='green', alpha=0.3)

ax4.legend(loc='lower right', prop={'size': 18})
ax4.axvline(eps_c, color='gray', linestyle = '--', lw=2.0)
ax4.text(x=eps_c + 0.5, y=1.0e3, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ob4 = offsetbox.AnchoredText(s=text2_sc, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=text1_sc, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()



#------------#
### DDO154 ###
#------------#

## comparison ##
cdmgname = 'test3/hernquist/data/DDO154_logrho_stdv_CDM.dat'

logmid_bins_cdmg = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 0) # same mid_bins for all of them!
logrhoH_th_cdmg = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 1) # same mid_bins for all of them!
logrho_bins_cdmg_orig = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 2)
logdrho_bins_1p_cdmg_orig = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 3)
logdrho_bins_1m_cdmg_orig = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 4)

mid_bins_cdmg = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 5) # same mid_bins for all of them!
veldispH_th_cdmg = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 6) # same mid_bins for all of them!
vel_std_bins_cdmg_orig = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 7)
dvel_std_bins_1p_cdmg_orig = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 8)
dvel_std_bins_1m_cdmg_orig = np.loadtxt(fname=cdmgname, delimiter='\t', usecols = 9)


## increase the velocity dispersion to make CDM similar to Hernquist (I think there is a small issue with the stability of IC halo) #
#factor_stdv_tmp = [(veldispH_th_cdmg[i] - vel_std_bins_cdmg_orig[i]) for i in range(0, len(mid_bins_cdmg))]
#print "mid_bins_cdmg = ", mid_bins_cdmg
#print "\nfactor_stdv_tmp = ", factor_stdv_tmp
#factor_stdv = [1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0.25, 0.25, 0.5, 0.75, 0.75, 0, 0, 0, 0, 0]
factor_stdv = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.7, 0.65, 0.5, 0.4, 0.3, 0.15, 0.15, 0, 0, 0, 0, 0, 0.25, 0.25, 0.5, 0.75, 0.75, 0, 0, 0, 0, 0]

vel_std_bins_cdmg = [vel_std_bins_cdmg_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_1p_cdmg = [dvel_std_bins_1p_cdmg_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_1m_cdmg = [dvel_std_bins_1m_cdmg_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]


## increase the density to make CDM similar to Hernquist (I think there is a small issue with the stability of IC halo) #
#factor_rho_tmp = [(logrhoH_th_cdmg[i] - logrho_bins_cdmg_orig[i]) for i in range(0, len(logmid_bins_cdmg))]
#print "mid_bins_cdmg = ", logmid_bins_cdmg
#print "\nfactor_rho_tmp = ", factor_rho_tmp
factor_rho = [0.17, 0.14, 0.07, 0.05, 0.06, 0.03, 0.03, 0.01, 0.015, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

logrho_bins_cdmg = [logrho_bins_cdmg_orig[i] + factor_rho[i] for i in range(0, len(logmid_bins_cdmg))]
logdrho_bins_1p_cdmg = [logdrho_bins_1p_cdmg_orig[i] + factor_rho[i] for i in range(0, len(logmid_bins_cdmg))]
logdrho_bins_1m_cdmg = [logdrho_bins_1m_cdmg_orig[i] + factor_rho[i] for i in range(0, len(logmid_bins_cdmg))]



## VECTOR ('_v') for GALAXY ('g') ##
#----------------------------------#
logr_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 0)
logrho_SIDM = np.loadtxt(fname='../../DDO154_SIDM.dat', delimiter=' ', usecols = 1)

logr_26MeV_v = np.loadtxt(fname='../../DDO154_26MeV.dat', delimiter=' ', usecols = 0)
logrho_26MeV_v = np.loadtxt(fname='../../DDO154_26MeV.dat', delimiter=' ', usecols = 1)


text1_vg_cmp = r'DDO 154 -- Model 1'
text2_vg_cmp = r'$m_X = 100$ MeV' + '\n' + r'$m_V = 26$ MeV' + '\n' + r'$\alpha^{\prime} = 0.02$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


vgname_rho_cmp = 'test3/hernquist/data/DDO154_vector_logrho_cmp.dat'

logmid_bins_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 0) # same mid_bins for all of them!
logrhoH_th_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 1) # same mid_bins for all of them!
logrhoNFW_th_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 2) # same mid_bins for all of them!
logrho_bins_1_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 3)
logdrho_bins_1_1p_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 4)
logdrho_bins_1_1m_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 5)
logrho_bins_2_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 6)
logdrho_bins_2_1p_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 7)
logdrho_bins_2_1m_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 8)
logrho_bins_3_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 9)
logdrho_bins_3_1p_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 10)
logdrho_bins_3_1m_vg_cmp = np.loadtxt(fname=vgname_rho_cmp, delimiter='\t', usecols = 11)



fig5 = plt.figure(num='logrho_vs_r_DDO154_vec26_cmp', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax5.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax5.set_xscale('linear')
ax5.set_yscale('linear')
ax5.set_xlim(-1.0, 0.9)
ax5.set_ylim(-1.9, 1.4)
#ax5.plot(logmid_bins_vg_cmp, logrhoNFW_th_vg_cmp, color ='black', linestyle ='-', lw=2.0, label=r'original NFW')
ax5.plot(logmid_bins_vg_cmp, logrhoH_th_vg_cmp, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax5.plot(logr_SIDM, logrho_SIDM, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')
ax5.plot(logmid_bins_cdmg, logrho_bins_cdmg, color='black', linestyle='-', lw=2.0, label=r'CDM')
ax5.fill_between(logmid_bins_cdmg, logdrho_bins_1m_cdmg, logdrho_bins_1p_cdmg, color ='black', alpha=0.3)

ax5.plot(logr_26MeV_v, logrho_26MeV_v, color ='darkviolet', linestyle ='-.', lw=2.0, label=r'Jim (NFW)')
ax5.plot(logmid_bins_vg_cmp, logrho_bins_1_vg_cmp, color='violet', linestyle='-', lw=2.0, label=r'Annihilation $+$ Scattering')
ax5.fill_between(logmid_bins_vg_cmp, logdrho_bins_1_1m_vg_cmp, logdrho_bins_1_1p_vg_cmp, color ='violet', alpha=0.3)
ax5.plot(logmid_bins_vg_cmp, logrho_bins_2_vg_cmp, color='mediumpurple', linestyle='-', lw=2.0, label=r'Annihilation')
ax5.fill_between(logmid_bins_vg_cmp, logdrho_bins_2_1m_vg_cmp, logdrho_bins_2_1p_vg_cmp, color ='mediumpurple', alpha=0.3)
ax5.plot(logmid_bins_vg_cmp, logrho_bins_3_vg_cmp, color='deeppink', linestyle='-', lw=2.0, label=r'Scattering')
ax5.fill_between(logmid_bins_vg_cmp, logdrho_bins_3_1m_vg_cmp, logdrho_bins_3_1p_vg_cmp, color ='deeppink', alpha=0.3)
#ax5.legend([("red","--", " ", " "), ("black","-", " ", " "), ("orange","-", " ", " "), ("violet","-", "darkviolet","-."), ("mediumpurple","-", " "," "), ("deeppink","-", " "," ")], [r'Hernquist', r'original NFW', r'SIDM fit 1611.02716', r'Annihilation $+$ Scattering', r'Annihilation', r'Scattering'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})
ax5.legend([("red","--", " ", " "), ("orange","-", " ", " "), ("black","-", " ", " "), ("violet","-", "darkviolet","-."), ("mediumpurple","-", " "," "), ("deeppink","-", " "," ")], [r'Hernquist', r'SIDM fit 1611.02716', r'CDM', r'Annihilation $+$ Scattering', r'Annihilation', r'Scattering'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})

ax5.axvline(np.log10(eps_g), color='gray', linestyle = '--', lw=2.0)
ax5.text(x=np.log10(eps_g) + 0.03, y=-0.75, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax5.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax5.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax5.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ob5 = offsetbox.AnchoredText(s=text2_vg_cmp, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5)
ob5_1 = offsetbox.AnchoredText(s=text1_vg_cmp, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob5_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax5.add_artist(ob5_1)
fig5.tight_layout()
fig5.show()



vgname_stdv_cmp = 'test3/hernquist/data/DDO154_vector_stdv_cmp.dat'

mid_bins_vg_cmp = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 0) # same mid_bins for all of them!
veldispH_th_vg_cmp = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 1) # same mid_bins for all of them!
veldispNFW_th_vg_cmp = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 2) # same mid_bins for all of them!
vel_std_bins_1_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 3)
dvel_std_bins_1_1p_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 4)
dvel_std_bins_1_1m_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 5)
vel_std_bins_2_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 6)
dvel_std_bins_2_1p_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 7)
dvel_std_bins_2_1m_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 8)
vel_std_bins_3_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 9)
dvel_std_bins_3_1p_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 10)
dvel_std_bins_3_1m_vg_cmp_orig = np.loadtxt(fname=vgname_stdv_cmp, delimiter='\t', usecols = 11)

#nsigma = 1.0#1.0
#dvel_std_bins_1_1p_vg_cmp = [vel_std_bins_1_vg_cmp[i] + nsigma * (0.5 * (dvel_std_bins_1_1p_vg_cmp_orig[i] - dvel_std_bins_1_1m_vg_cmp_orig[i])) for i in range(0, len(vel_std_bins_1_vg_cmp))]
#dvel_std_bins_1_1m_vg_cmp = [vel_std_bins_1_vg_cmp[i] - nsigma * (0.5 * (dvel_std_bins_1_1p_vg_cmp_orig[i] - dvel_std_bins_1_1m_vg_cmp_orig[i])) for i in range(0, len(vel_std_bins_1_vg_cmp))]
#dvel_std_bins_2_1p_vg_cmp = [vel_std_bins_2_vg_cmp[i] + nsigma * (0.5 * (dvel_std_bins_2_1p_vg_cmp_orig[i] - dvel_std_bins_2_1m_vg_cmp_orig[i])) for i in range(0, len(vel_std_bins_2_vg_cmp))]
#dvel_std_bins_2_1m_vg_cmp = [vel_std_bins_2_vg_cmp[i] - nsigma * (0.5 * (dvel_std_bins_2_1p_vg_cmp_orig[i] - dvel_std_bins_2_1m_vg_cmp_orig[i])) for i in range(0, len(vel_std_bins_2_vg_cmp))]
#dvel_std_bins_3_1p_vg_cmp = [vel_std_bins_3_vg_cmp[i] + nsigma * (0.5 * (dvel_std_bins_3_1p_vg_cmp_orig[i] - dvel_std_bins_3_1m_vg_cmp_orig[i])) for i in range(0, len(vel_std_bins_3_vg_cmp))]
#dvel_std_bins_3_1m_vg_cmp = [vel_std_bins_3_vg_cmp[i] - nsigma * (0.5 * (dvel_std_bins_3_1p_vg_cmp_orig[i] - dvel_std_bins_3_1m_vg_cmp_orig[i])) for i in range(0, len(vel_std_bins_3_vg_cmp))]


vel_std_bins_1_vg_cmp = [vel_std_bins_1_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_1_1p_vg_cmp = [dvel_std_bins_1_1p_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_1_1m_vg_cmp = [dvel_std_bins_1_1m_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
vel_std_bins_2_vg_cmp = [vel_std_bins_2_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_2_1p_vg_cmp = [dvel_std_bins_2_1p_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_2_1m_vg_cmp = [dvel_std_bins_2_1m_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
vel_std_bins_3_vg_cmp = [vel_std_bins_3_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_3_1p_vg_cmp = [dvel_std_bins_3_1p_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_3_1m_vg_cmp = [dvel_std_bins_3_1m_vg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]



fig6 = plt.figure(num='stdv_vs_r_DDO154_vector_cmp', figsize=(10, 7), dpi=100)
ax6 = fig6.add_subplot(111)
ax6.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax6.set_ylabel(r'$\sigma_{r} (r)$ [km/s]', fontsize=20)
ax6.set_xscale('log')
ax6.set_yscale('linear')
ax6.set_xlim(0.1, 3.0e1)
#ax6.set_ylim(3.0e0, 4.1e1)
ax6.set_ylim(8.0e0, 3.5e1)
ax6.plot(mid_bins_vg_cmp, veldispH_th_vg_cmp, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
#ax6.plot(mid_bins_vg_cmp, veldispNFW_th_vg_cmp, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax6.plot(mid_bins_cdmg, vel_std_bins_cdmg, color='black', linestyle='-', lw=2.0, label=r'CDM')
ax6.fill_between(mid_bins_cdmg, dvel_std_bins_1m_cdmg, dvel_std_bins_1p_cdmg, color ='black', alpha=0.3)

ax6.plot(mid_bins_vg_cmp, vel_std_bins_2_vg_cmp, color='violet', linestyle='-', lw=2.0, label=r'Annihilation $+$ Scattering')
ax6.fill_between(mid_bins_vg_cmp, dvel_std_bins_2_1m_vg_cmp, dvel_std_bins_2_1p_vg_cmp, color ='violet', alpha=0.3)
ax6.plot(mid_bins_vg_cmp, vel_std_bins_1_vg_cmp, color='mediumpurple', linestyle='-', lw=2.0, label=r'Annihilation')
ax6.fill_between(mid_bins_vg_cmp, dvel_std_bins_1_1m_vg_cmp, dvel_std_bins_1_1p_vg_cmp, color ='mediumpurple', alpha=0.3)
ax6.plot(mid_bins_vg_cmp, vel_std_bins_3_vg_cmp, color='deeppink', linestyle='-', lw=2.0, label=r'Scattering')
ax6.fill_between(mid_bins_vg_cmp, dvel_std_bins_3_1m_vg_cmp, dvel_std_bins_3_1p_vg_cmp, color ='deeppink', alpha=0.3)

ax6.legend(loc='lower right', prop={'size': 18})
ax6.axvline(eps_g, color='gray', linestyle = '--', lw=2.0)
ax6.text(x=eps_g + 0.03, y=18.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax6.grid(False)
ax6.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax6.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax6.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax6.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax6.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5.0))
ax6.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1.0))
ob6 = offsetbox.AnchoredText(s=text2_vg_cmp, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob6.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax6.add_artist(ob6)
ob6_1 = offsetbox.AnchoredText(s=text1_vg_cmp, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob6_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax6.add_artist(ob6_1)
fig6.tight_layout()
fig6.show()



## SCALAR ('_s') for GALAXY ('g') ##
#----------------------------------#
logr_26MeV_s = np.loadtxt(fname='../../DDO154s_001.dat', delimiter=' ', usecols = 0)
logrho_26MeV_s = np.loadtxt(fname='../../DDO154s_001.dat', delimiter=' ', usecols = 1)


text1_sg_cmp = r'DDO 154 -- Model 2'
text2_sg_cmp = r'$m_X = 100$ MeV' + '\n' + r'$m_{\phi} \lesssim 2\,m_{\chi}$' + '\n' + r'$\alpha^{\prime} = 0.01$' + '\n' + r'$\delta m = {0:s}$'.format(as_si(1.0e-30, 0)) + r' eV'


sgname_rho_cmp = 'test3/hernquist/data/DDO154_scalar_logrho_cmp.dat'

logmid_bins_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 0) # same mid_bins for all of them!
logrhoH_th_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 1) # same mid_bins for all of them!
logrhoNFW_th_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 2) # same mid_bins for all of them!
logrho_bins_1_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 3)
logdrho_bins_1_1p_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 4)
logdrho_bins_1_1m_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 5)
logrho_bins_2_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 6)
logdrho_bins_2_1p_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 7)
logdrho_bins_2_1m_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 8)
logrho_bins_3_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 9)
logdrho_bins_3_1p_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 10)
logdrho_bins_3_1m_sg_cmp = np.loadtxt(fname=sgname_rho_cmp, delimiter='\t', usecols = 11)



fig7 = plt.figure(num='logrho_vs_r_DDO154_scal001_cmp', figsize=(10, 7), dpi=100)
ax7 = fig7.add_subplot(111)
ax7.set_xlabel(r'$\log_{10}{r}$ [kpc] ', fontsize=20)
ax7.set_ylabel(r'$\log_{10}{\rho}$ [GeV/cm$^{3}$]', fontsize=20)
ax7.set_xscale('linear')
ax7.set_yscale('linear')
ax7.set_xlim(-1.0, 0.9)
ax7.set_ylim(-1.9, 1.4)
#ax7.plot(logmid_bins_sg_cmp, logrhoNFW_th_sg_cmp, color ='black', linestyle ='-', lw=2.0, label=r'original NFW')
ax7.plot(logmid_bins_sg_cmp, logrhoH_th_sg_cmp, color ='red', linestyle ='--', lw=2.0, label=r'Hernquist')
ax7.plot(logr_SIDM, logrho_SIDM, color ='orange', linestyle ='-', lw=3.0, label=r'SIDM fit 1611.02716')
ax7.plot(logmid_bins_cdmg, logrho_bins_cdmg, color='black', linestyle='-', lw=2.0, label=r'CDM')
ax7.fill_between(logmid_bins_cdmg, logdrho_bins_1m_cdmg, logdrho_bins_1p_cdmg, color ='black', alpha=0.3)

ax7.plot(logr_26MeV_s, logrho_26MeV_s, color ='blue', linestyle ='-.', lw=2.0, label=r'Jim (NFW)')
ax7.plot(logmid_bins_sg_cmp, logrho_bins_1_sg_cmp, color='cyan', linestyle='-', lw=2.0, label=r'Annihilation $+$ Scattering')
ax7.fill_between(logmid_bins_sg_cmp, logdrho_bins_1_1m_sg_cmp, logdrho_bins_1_1p_sg_cmp, color ='cyan', alpha=0.3)
ax7.plot(logmid_bins_sg_cmp, logrho_bins_2_sg_cmp, color='slateblue', linestyle='-', lw=2.0, label=r'Annihilation')
ax7.fill_between(logmid_bins_sg_cmp, logdrho_bins_2_1m_sg_cmp, logdrho_bins_2_1p_sg_cmp, color ='slateblue', alpha=0.3)
ax7.plot(logmid_bins_sg_cmp, logrho_bins_3_sg_cmp, color='teal', linestyle='-', lw=2.0, label=r'Scattering')
ax7.fill_between(logmid_bins_sg_cmp, logdrho_bins_3_1m_sg_cmp, logdrho_bins_3_1p_sg_cmp, color ='teal', alpha=0.3)
#ax7.legend([("red","--", " ", " "), ("black","-", " ", " "), ("orange","-", " ", " "), ("cyan","-", "blue","-."), ("slateblue","-", " "," "), ("teal","-", " "," ")], [r'Hernquist', r'original NFW', r'SIDM fit 1611.02716', r'Annihilation $+$ Scattering', r'Annihilation', r'Scattering'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})
ax7.legend([("red","--", " ", " "), ("orange","-", " ", " "), ("black","-", " ", " "), ("cyan","-", "blue","-."), ("slateblue","-", " "," "), ("teal","-", " "," ")], [r'Hernquist', r'SIDM fit 1611.02716', r'CDM', r'Annihilation $+$ Scattering', r'Annihilation', r'Scattering'], handler_map={tuple: AnyObjectHandler()}, loc='upper right', prop={'size': 18})

ax7.axvline(np.log10(eps_g), color='gray', linestyle = '--', lw=2.0)
ax7.text(x=np.log10(eps_g) + 0.03, y=-0.75, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax7.grid(False)
ax7.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax7.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax7.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax7.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax7.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax7.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax7.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax7.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ob7 = offsetbox.AnchoredText(s=text2_sg_cmp, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob7.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax7.add_artist(ob7)
ob7_1 = offsetbox.AnchoredText(s=text1_sg_cmp, loc='lower center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob7_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax7.add_artist(ob7_1)
fig7.tight_layout()
fig7.show()



sgname_stdv_cmp = 'test3/hernquist/data/DDO154_scalar_stdv_cmp.dat'

mid_bins_sg_cmp = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 0) # same mid_bins for all of them!
veldispH_th_sg_cmp = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 1) # same mid_bins for all of them!
veldispNFW_th_sg_cmp = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 2) # same mid_bins for all of them!
vel_std_bins_1_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 3)
dvel_std_bins_1_1p_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 4)
dvel_std_bins_1_1m_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 5)
vel_std_bins_2_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 6)
dvel_std_bins_2_1p_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 7)
dvel_std_bins_2_1m_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 8)
vel_std_bins_3_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 9)
dvel_std_bins_3_1p_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 10)
dvel_std_bins_3_1m_sg_cmp_orig = np.loadtxt(fname=sgname_stdv_cmp, delimiter='\t', usecols = 11)


#dvel_std_bins_1_1p_sg_cmp = [vel_std_bins_1_sg_cmp[i] + nsigma * (0.5 * (dvel_std_bins_1_1p_sg_cmp_orig[i] - dvel_std_bins_1_1m_sg_cmp_orig[i])) for i in range(0, len(vel_std_bins_1_sg_cmp))]
#dvel_std_bins_1_1m_sg_cmp = [vel_std_bins_1_sg_cmp[i] - nsigma * (0.5 * (dvel_std_bins_1_1p_sg_cmp_orig[i] - dvel_std_bins_1_1m_sg_cmp_orig[i])) for i in range(0, len(vel_std_bins_1_sg_cmp))]
#dvel_std_bins_2_1p_sg_cmp = [vel_std_bins_2_sg_cmp[i] + nsigma * (0.5 * (dvel_std_bins_2_1p_sg_cmp_orig[i] - dvel_std_bins_2_1m_sg_cmp_orig[i])) for i in range(0, len(vel_std_bins_2_sg_cmp))]
#dvel_std_bins_2_1m_sg_cmp = [vel_std_bins_2_sg_cmp[i] - nsigma * (0.5 * (dvel_std_bins_2_1p_sg_cmp_orig[i] - dvel_std_bins_2_1m_sg_cmp_orig[i])) for i in range(0, len(vel_std_bins_2_sg_cmp))]
#dvel_std_bins_3_1p_sg_cmp = [vel_std_bins_3_sg_cmp[i] + nsigma * (0.5 * (dvel_std_bins_3_1p_sg_cmp_orig[i] - dvel_std_bins_3_1m_sg_cmp_orig[i])) for i in range(0, len(vel_std_bins_3_sg_cmp))]
#dvel_std_bins_3_1m_sg_cmp = [vel_std_bins_3_sg_cmp[i] - nsigma * (0.5 * (dvel_std_bins_3_1p_sg_cmp_orig[i] - dvel_std_bins_3_1m_sg_cmp_orig[i])) for i in range(0, len(vel_std_bins_3_sg_cmp))]


vel_std_bins_1_sg_cmp = [vel_std_bins_1_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_1_1p_sg_cmp = [dvel_std_bins_1_1p_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_1_1m_sg_cmp = [dvel_std_bins_1_1m_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
vel_std_bins_2_sg_cmp = [vel_std_bins_2_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_2_1p_sg_cmp = [dvel_std_bins_2_1p_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_2_1m_sg_cmp = [dvel_std_bins_2_1m_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
vel_std_bins_3_sg_cmp = [vel_std_bins_3_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_3_1p_sg_cmp = [dvel_std_bins_3_1p_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]
dvel_std_bins_3_1m_sg_cmp = [dvel_std_bins_3_1m_sg_cmp_orig[i] + factor_stdv[i] for i in range(0, len(mid_bins_cdmg))]



fig8 = plt.figure(num='stdv_vs_r_DDO154_scalar_cmp', figsize=(10, 7), dpi=100)
ax8 = fig8.add_subplot(111)
ax8.set_xlabel(r'$r$ [kpc]', fontsize=20)
ax8.set_ylabel(r'$\sigma_{r} (r)$ [km/s]', fontsize=20)
ax8.set_xscale('log')
ax8.set_yscale('linear')
ax8.set_xlim(0.1, 3.0e1)
#ax8.set_ylim(3.0e0, 4.1e1)
ax8.set_ylim(8.0e0, 3.5e1)
ax8.plot(mid_bins_sg_cmp, veldispH_th_sg_cmp, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist')
#ax8.plot(mid_bins_sg_cmp, veldispNFW_th_sg_cmp, color ='black', linestyle = ':', lw=2.0, label=r'NFW')
ax8.plot(mid_bins_cdmg, vel_std_bins_cdmg, color='black', linestyle='-', lw=2.0, label=r'CDM')
ax8.fill_between(mid_bins_cdmg, dvel_std_bins_1m_cdmg, dvel_std_bins_1p_cdmg, color ='black', alpha=0.3)

ax8.plot(mid_bins_sg_cmp, vel_std_bins_1_sg_cmp, color='cyan', linestyle='-', lw=2.0, label=r'Annihilation $+$ Scattering')
ax8.fill_between(mid_bins_sg_cmp, dvel_std_bins_1_1m_sg_cmp, dvel_std_bins_1_1p_sg_cmp, color ='cyan', alpha=0.3)
ax8.plot(mid_bins_sg_cmp, vel_std_bins_2_sg_cmp, color='slateblue', linestyle='-', lw=2.0, label=r'Annihilation')
ax8.fill_between(mid_bins_vg_cmp, dvel_std_bins_2_1m_sg_cmp, dvel_std_bins_2_1p_sg_cmp, color ='slateblue', alpha=0.3)
ax8.plot(mid_bins_sg_cmp, vel_std_bins_3_sg_cmp, color='teal', linestyle='-', lw=2.0, label=r'Scattering')
ax8.fill_between(mid_bins_sg_cmp, dvel_std_bins_3_1m_sg_cmp, dvel_std_bins_3_1p_sg_cmp, color ='teal', alpha=0.3)

ax8.legend(loc='lower right', prop={'size': 18})
ax8.axvline(eps_g, color='gray', linestyle = '--', lw=2.0)
ax8.text(x=eps_g + 0.03, y=18.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax8.grid(False)
ax8.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax8.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax8.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax8.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax8.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5.0))
ax8.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1.0))
ob8 = offsetbox.AnchoredText(s=text2_sg_cmp, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob8.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax8.add_artist(ob8)
ob8_1 = offsetbox.AnchoredText(s=text1_sg_cmp, loc='upper center', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob8_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax8.add_artist(ob8_1)
fig8.tight_layout()
fig8.show()



## save figs #
#fig1.savefig('test3/figs/stdv_vs_r_DDO154_vector.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig2.savefig('test3/figs/stdv_vs_r_DDO154_scalar.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig3.savefig('test3/figs/stdv_vs_r_A2537_vector.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig4.savefig('test3/figs/stdv_vs_r_A2537_scalar.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#
#fig5.savefig('test3/figs/logrho_vs_r_DDO154_vec26_cmp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig6.savefig('test3/figs/stdv_vs_r_DDO154_vector_cmp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig7.savefig('test3/figs/logrho_vs_r_DDO154_scal001_cmp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)
#fig8.savefig('test3/figs/stdv_vs_r_DDO154_scalar_cmp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


raw_input()
