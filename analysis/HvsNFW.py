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

#print "rhocrit = ", rhocrit
#sys.exit()

vbar = 200.0


##-- NFW --##
def M200(r200) :
	return 4.0 * np.pi / 3.0 * r200**3 * vbar * rhocrit 

def R200(m200) :
	r3 = 3.0 * m200 / (4.0 * np.pi * vbar * rhocrit)
	return r3**(1.0/3.0)


def gc(c) :
	tmp = np.log(1.0 + c) - c / (1.0 + c)
	return 1.0 / tmp

def rhos_cte(c) :
	return 200.0 / 3.0 * rhocrit * c**3 * gc(c)

def C2(c, Mv, rv) :
	return Mv * c**2 * gc(c) / (4.0 * np.pi * rv**3)

def rhos(s, c, Mv, rv) :
	x = c * s
	return C2(c, Mv, rv) * c / (x * (1.0 + x)**2)

def veldisps(s, c, Vv) :
	x = c * s
	arg = np.pi**2 - np.log(x) - 1.0 / x - 1.0 / (1.0 + x)**2 - 6.0 / (1.0 + x) + (1.0 + 1.0 / x**2 - 4.0 / x - 2.0 / (1.0 + x)) * np.log(1.0 + x) + 3.0 * (np.log(1.0 + x))**2 + 6.0 * spence(1.0 + x)
	tmp = 0.5 * c**2 * gc(c) * s * (1.0 + x)**2 * arg
	if tmp < 0 :
		tmp = 0
	return Vv * np.sqrt(tmp)


def rhoNFW(r, rhoss, rs) :
	return rhoss / (r / rs) / (1.0 + r / rs)**2

def M200s(rhoss, rs, c) :
	return 4.0 * np.pi * rhoss * rs**3 / gc(c)


##-- Hernquist --##
def rhoH(r, M, a) :
	return M / (2.0 * np.pi) * (a / r) / (r + a)**3

def veldispH(r, M, a) :
	fH = 12.0 * r * (r + a)**3 / a**4 * np.log((r + a)/r) - r/(r + a) * (25.0 + 52.0 * (r / a) + 42.0 * (r / a)**2 + 12.0 * (r / a)**3)
	return np.sqrt(G * M / (12.0 * a) * fH)



def aHvsNFW(r200, c) :
	return r200 / (np.sqrt(c**2 / 2.0 / (np.log(1.0 + c) - c / (1.0 + c))) - 1.0)

def MHvsNFW(r200, c) :
	a = aHvsNFW(r200, c)
	m200 = M200(r200)
	return m200 * (r200 + a)**2 / r200**2

def aHvsNFW_Springel(r200, c) :
	return r200 / c * np.sqrt( 2.0 * (np.log(1.0 + c) - c / (1.0 + c)) )


print "----- KODA ------"
# Jeremie's paper
Msun = 1.98841e33 # g
kpc = 3.08567758149e15 # cm

c_1 = 5.722240632115952 # required to get the same rs and rhos_0 he used
m200_1 = 2.731387826049509
rs = R200(m200_1)/c_1
rhos_0 = 200.0 * (c_1)**3 * gc(c_1) * rhocrit / 3.0
print "c = ", c_1
print "m200 = ", m200_1 # 1e10 Msun
print "r200 = ", R200(m200_1) # kpc
print "rs = ", rs # kpc
print "rhos_0 = ", rhos_0 # 1e10 Msun/kpc^3
print "vs = ", np.sqrt(4.0 * np.pi * G * rhos_0) * rs # km/s
print "1/(rhos_0 * rs) = ", 1.0 / (rhos_0 * rs) * kpc**2 / (1.0e10 * Msun) # cm^2 / g

# Koda : r_max = 100 rs
def Ms(rors, m200, c) :
	return m200 * gc(c) * (np.log(1.0 + rors) - rors / (1.0 + rors))

print "Ms = ", Ms(100.0, m200_1, c_1)
print "-----------------"



# NFW and simulation parameters in code units!

##---------# (DDO 154 dSph galaxy) #---------##
Mv1 = 2.3 # * 1e10 Msun (total mass -> dSph galaxy)
rv1 = R200(Mv1) # kpc
rhos1 = 1.5e-3 # * 1e10 Msun/kpc^3
rs1 = 3.4 # kpc
Vv1 = 49.0 # km/s
c1 = 12.2
eps = 0.3 # kpc

###---------# (A 2537 galaxy cluster) #---------##
#rv1 = 2050.0 # kpc
#Mv1 = M200(rv1) # * 1e10 Msun (total mass -> galaxy cluster)
#rhos1 = 1.3e-4 # * 1e10 Msun/kpc^3
#rs1 = 442.0 # kpc
#Vv1 = 1660.0 # km/s
#c1 = 4.63
#eps = 9.3 # kpc


###---------# (DDO 126 dSph galaxy) #---------##
#Mv1 = 1.02 # * 1e10 Msun (total mass -> dSph galaxy)
#rv1 = R200(Mv1) # kpc
#rhos1 = 1.138e-3 # * 1e10 Msun/kpc^3
#rs1 = 3.455 # kpc
#Vv1 = np.sqrt(G * Mv1 / rv1) # km/s
#c1 = 13.29
#eps = 0.23 # kpc



Mv1_original = Mv1
rv1_original = rv1

# inferred Mv1, rv1 using rhos1, rs1, c1 #
Mv1 = M200s(rhoss=rhos1, rs=rs1, c=c1)
rv1 = c1 * rs1


##-- Max radius in units of rs --##
rOrs_max = 1.0e3 # cut-off in the radius
time = 0.0

r_max = rOrs_max * rs1
r_min = 0.01 # kpc


#---- WE USE RHOS1, RS1 and C1 TO DESCRIBE HALO ----#
##-- find M, a Hernquist parameters (Robertson) --##
M1 = MHvsNFW(r200=rv1, c=c1)
a1 = aHvsNFW(r200=rv1, c=c1)
# Springel (https://arxiv.org/pdf/astro-ph/0411108.pdf)
M2 = Mv1
a2 = aHvsNFW_Springel(r200=rv1, c=c1) #--> better Springel


print "\nMv1_original = ", Mv1_original
print "\nMv = ", Mv1
print "rv = ", rv1
print "c = ", c1
print "\nM (Robertson) = ", M1
print "a (Robertson) = ", a1
print "\nM (Springel) = ", M2
print "a (Springel) = ", a2

print "\nVv = ", np.sqrt(G * Mv1 / rv1)


bins_new = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=1e4)

## NFW ##
rhoNFW_th = [rhoNFW(r=el, rhoss=rhos1*1.0e10, rs=rs1) for el in bins_new]
rhoNFW_th_alt = [rhos(s=el/rv1, c=c1, Mv=Mv1*1.0e10, rv=rv1) for el in bins_new]

veldispNFW_th = [veldisps(s=el/rv1, c=c1, Vv=Vv1) for el in bins_new] # G*M has right units

## Hernquist ##
rhoH_th = [rhoH(r=el, M=M1*1.0e10, a=a1) for el in bins_new]
veldispH_th = [veldispH(r=el, M=M1, a=a1) for el in bins_new]

rhoH_th_Springel = [rhoH(r=el, M=M2*1.0e10, a=a2) for el in bins_new]
veldispH_th_Springel = [veldispH(r=el, M=M2, a=a2) for el in bins_new]


#text1 = r'$M_v = {0:s}$'.format(as_si(Mv1*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1) + r' kpc' + '\n' + r'$M_R = {0:s}$'.format(as_si(M1*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$a_R = {0:0.1f}$'.format(a1) + r' kpc' + '\n' + r'$M_S = {0:s}$'.format(as_si(M2*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$a_S = {0:0.1f}$'.format(a2) + r' kpc'
text1 = r'$M_v = {0:s}$'.format(as_si(Mv1_original*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$r_v = {0:0.1f}$'.format(rv1_original) + r' kpc' + '\n' + r'$M = {0:s}$'.format(as_si(M2*1.0e10, 1)) + r' M$_{\odot}$' + '\n' + r'$a = {0:0.1f}$'.format(a2) + r' kpc'


fig3 = plt.figure(num='stdv_vs_r', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$r$ [kpc] ', fontsize=20)
ax3.set_ylabel(r'$\sigma_{v_r} (r)$ [km/s]', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(r_min, r_max)
ax3.set_ylim(2.0, 1.0e2)
ax3.plot(bins_new, veldispNFW_th, color ='black', linestyle = '-', lw=2.0, label=r'NFW')
ax3.plot(bins_new, veldispH_th_Springel, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist (Springel)')
ax3.plot(bins_new, veldispH_th, color ='green', linestyle = '-.', lw=2.0, label=r'Hernquist (Robertson)')
ax3.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax3.text(x=eps + 0.2, y=10.0, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax3.axvspan(r_min, rv1, color='orange', alpha=0.2)
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='upper right', prop={'size': 18})
ob3 = offsetbox.AnchoredText(s=r'$t = $ ' + str(np.around(a=time, decimals=1)) + r' Gyr', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob3.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax3.add_artist(ob3)
ob3_1 = offsetbox.AnchoredText(s=text1, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
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
ax4.set_xlim(r_min, r_max)
#ax4.set_ylim(1.0e-1, 1.0e10)
ax4.plot(bins_new, rhoNFW_th, color ='black', linestyle = '-', lw=2.0, label=r'NFW')
ax4.plot(bins_new, rhoNFW_th_alt, color ='cyan', linestyle = ':', lw=2.0, label=r'NFW alternative')
ax4.plot(bins_new, rhoH_th_Springel, color ='red', linestyle = '--', lw=2.0, label=r'Hernquist (Springel)')
ax4.plot(bins_new, rhoH_th, color ='green', linestyle = '-.', lw=2.0, label=r'Hernquist (Robertson)')
ax4.axvline(eps, color='gray', linestyle = '-.', lw=2.0)
ax4.text(x=eps + 0.2, y=2.0e3, s=r'$\epsilon$', rotation=0, color='gray', fontsize=18)
ax4.axvspan(r_min, rv1, color='orange', alpha=0.2)
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper right', prop={'size': 18})
ob4 = offsetbox.AnchoredText(s=r'$t = $ ' + str(np.around(a=time, decimals=1)) + r' Gyr', loc='lower right', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4)
ob4_1 = offsetbox.AnchoredText(s=text1, loc='lower left', pad=0.3, borderpad=0.85, frameon=True, prop=dict(color='black', size=18))
ob4_1.patch.set(boxstyle='round', edgecolor='lightgray', alpha=1.0)
ax4.add_artist(ob4_1)
fig4.tight_layout()
fig4.show()



raw_input()



