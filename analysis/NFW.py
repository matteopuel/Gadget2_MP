import numpy as np
from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from itertools import chain
import struct
import pygadgetreader as pg

#from scipy.special import lambertw
from pynverse import inversefunc, piecewise
from scipy import interpolate, integrate


plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 18})
rc('text', usetex=True)
#rc('font', size=17)
#rc('legend', fontsize=15)

rc('font', family='serif', size=18)
rc('legend', fontsize=18)
#plt.style.use('classic')


# IC generation: isolated halo with Hernquist DM density profile, matched with a NFW profile


##-- DATA --#
# GADGET-2 code units
UnitLength_in_cm = 3.085678e21  # = 1 kpc
UnitMass_in_g = 1.989e43  # = 1e10 Msun
UnitVelocity_in_cm_per_s = 1e5  # = 1 km/s
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # in sec -> = 0.979 Gyr


# constants from simulation or PDG
h = 0.6732117
Gyr = (365*24*60*60)*1.0e9 # sec
rhocrit = 1.87834e-29 * h**2 # g/cm^3
rhocrit *= (UnitLength_in_cm**3 / UnitMass_in_g) # 1e10 Msun / kpc^3 (code units) 
G = 6.672e-8 # cm^3/g/s^2
G *= (UnitMass_in_g * UnitTime_in_s**2 / UnitLength_in_cm**3) # (code units)

vbar = 200.0

def M200(r200) :
	return 4.0 * np.pi / 3.0 * r200**3 * vbar * rhocrit 

def R200(m200) :
	r3 = 3.0 * m200 / (4.0 * np.pi * vbar * rhocrit)
	return r3**(1.0/3.0)


def gc(c) :
	tmp = np.log(1.0 + c) - c / (1.0 + c)
	return 1.0 / tmp

def C1(c, Vv) :
	return Vv**2 * gc(c)

def Psi(s, c, Vv) : # = - Phi(s, c, Vv)
	# s = r / rv, but c = rv / rs
	x = c * s
	if x == 0 :
		return C1(c, Vv) * c
	else :
		return C1(c, Vv) * c * np.log(1.0 + x) / x

def C2(c, Mv=False, rv=False) :
	if Mv == False and rv != False :
		Mv = M200(rv)
	elif Mv != False and rv == False :
		rv = R200(Mv)
	else :
		print "Mv and rv are (not) given!"
		sys.exit()

	return Mv * c**2 * gc(c) / (4.0 * np.pi * rv**3)

def rhos(s, c, Mv=False, rv=False) :
	return C2(c, Mv, rv) / (s * (1.0 + c * s)**2)

def sPsi(psi, c, Vv, approx=False) :
	if approx == False :
		Psifunc = lambda s: piecewise(s, [s < 0, s >= 0], [lambda s: 0, lambda s: Psi(s, c, Vv)])
		#Psifunc = lambda s: Psi(s, c, Vv)
		func = inversefunc(Psifunc, accuracy=4)
		return func(psi).tolist()
	else :
		psi_tilde = psi / C1(c, Vv)
		return - 1.75 * np.log(psi_tilde / c) / psi_tilde

def rhopsi(psi, c, Vv, Mv=False, rv=False, approx=False) :
	spsi = sPsi(psi, c, Vv, approx)
	return rhos(spsi, c, Mv, rv)


## derivatives ##
def dPsi(s, c, Vv) :
	x = c * s
	return - Psi(s, c, Vv) * c / x + C1(c, Vv) * c**3 / x**2 / (1.0 + x)

def drhos(s, c, Mv=False, rv=False) :
	x = c * s
	return - C2(c, Mv, rv) * c**2 * (1.0 + 3.0 * x) / x**2 / (1.0 + x)**3

def ddrhos(s, c, Mv=False, rv=False) :
	x = c * s
	return 2.0 * C2(c, Mv, rv) * c**3 * (1.0 + 4.0 * x + 6.0 * x**2) / x**3 / (1.0 + x)**4



# Model parameters in code units #
##---------# (DDO 154 dSph galaxy) #---------#
Mv1 = 2.3 # * 1e10 Msun (total mass -> dSph galaxy)
rv1 = R200(Mv1) # kpc
rhos1 = 1.5e-3 # * 1e10 Msun/kpc^3
rs = 3.4 # kpc
Vv1 = 49.0 # km/s
c1 = 12.2
str_a = "nfw_test3_dsph.dat" #### it does not work!!!!

## import Mathematica points for approximate solution for same model parameters: from "NFW.nb" in ADMgal directory ##
ftab_eps_approx =  np.loadtxt(fname='../../ftab.dat', delimiter='\t', usecols = 0)
ftab_feps_approx = np.loadtxt(fname='../../ftab.dat', delimiter='\t', usecols = 1)


###---------# (A2537 galaxy cluster) #---------#
#rv1 = 2050.0 # kpc
#Mv1 = M200(rv1) # * 1e10 Msun (total mass -> galaxy cluster)
#rhos1 = 1.3e-4 # * 1e10 Msun/kpc^3
#rs = 442 # kpc
#Vv1 = 1660.0 # km/s
#c1 = 4.63
#str_a = "nfw_test3_cluster.dat"

### import Mathematica points for approximate solution for same model parameters: from "NFW.nb" in ADMgal directory ##
#ftab_eps_approx =  np.loadtxt(fname='../../ftab_c.dat', delimiter='\t', usecols = 0)
#ftab_feps_approx = np.loadtxt(fname='../../ftab_c.dat', delimiter='\t', usecols = 1)


fEunit = np.sqrt(8.0) * Mv1 / (rv1*Vv1)**3 # as in Figure 5 in Lokas & Mamon 2001
Eunit = Vv1**2



s_list = np.logspace(start=np.log10(4.5e-5), stop=np.log10(4.5e9),num=1e4) # s = r / rs / c ## better!
#s_list = np.linspace(start=1.0e-8, stop=1.0e2,num=1e4) # s = r / rs / c



## check accuracy of sPsi ## --> ok, 4th digit accuracy!
Psi_list = [Psi(el, c=c1, Vv=Vv1) for el in s_list]
sPsi_list = [sPsi(el, c=c1, Vv=Vv1, approx=False) for el in Psi_list]
sPsi_approx_list = [sPsi(el, c=c1, Vv=Vv1, approx=True) for el in Psi_list]

Psi_check_list = [Psi(sPsi(el, c=c1, Vv=Vv1, approx=False), c=c1, Vv=Vv1) for el in Psi_list]
print "\nPsi_list[0:10] = ", Psi_list[0:10]
print "Psi_check_list[0:10] = ", Psi_check_list[0:10]
print "\ns_list[0:10] = ", s_list[0:10]
print "s_check_list[0:10] = ", sPsi_list[0:10]


def sPsi_Math(psi, c, Vv) :
	arg = psi / C1(c, Vv)
	return - 1.75 * np.log(arg / c) / arg
	#return (- 1.75 * Vv**2 * np.log((psi * (- (c / (1.0 + c)) + np.log(1.0 + c))) / (c * Vv**2))) / (psi * (- (c / (1.0 + c)) + np.log(1.0 + c)))

sPsi_Math_list = [sPsi_Math(el, c=c1, Vv=Vv1) for el in Psi_list]


fig0 = plt.figure(num='sPsi', figsize=(10, 7), dpi=100)
ax0 = fig0.add_subplot(111)
ax0.set_xlabel(r'$\Psi$', fontsize=20)
ax0.set_ylabel(r'$s$', fontsize=20)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.plot(Psi_list, sPsi_list, color ='red', linestyle = '-', lw=2.0, label=r'exact python')
ax0.plot(Psi_list, sPsi_approx_list, color ='orange', linestyle = '--', lw=2.0, label=r'approx python')
ax0.plot(Psi_list, sPsi_Math_list, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
ax0.grid(False)
ax0.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax0.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax0.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax0.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax0.legend(loc='lower left', prop={'size': 18})
fig0.tight_layout()
fig0.show()


#sPsi_interp = interpolate.InterpolatedUnivariateSpline(x=Psi_list[::-1], y=sPsi_list[::-1], k=3)
#sPsi_p_func = sPsi_interp.derivative(n=1)
#sPsi_pp_func = sPsi_interp.derivative(n=2)
#
#sPsi_p_func_list = sPsi_p_func(Psi_list) # s_list
#sPsi_pp_func_list = sPsi_pp_func(Psi_list)
#
#ddrhopsi_list = []
#for i in range(0, len(s_list)) :
#	tmp = sPsi_p_func_list[i]**2 * ddrhos(s_list[i], c=c1, Mv=Mv1) + sPsi_pp_func_list[i] * drhos(s_list[i], c=c1, Mv=Mv1)
#	ddrhopsi_list.append(tmp)
#
#ddrhopsi_interp = interpolate.InterpolatedUnivariateSpline(x=s_list, y=ddrhopsi_list, k=3)
#ddrhopsi_interp_list = [ddrhopsi_interp(el) for el in s_list]
#
#
#def ddrhopsi_Math(s, c, Vv, Mv=False, rv=False) :
#	x = c * s
#	log1 = np.log(1.0 + x) / x
#	loglog = np.log(log1)
#	num = (1.0 + x) * (1.0 + 3.0 * x) * np.log(1.0 + x) * (5.25 - 3.5 * loglog) + 2.0 * (1.0 + 4.0 * x + 6.0 * x**2) * 3.0625 * (1.0 - loglog)**2
#	den = C1(c, Vv)**2 * (1.0 + x)**4 * np.log(1.0 + x)**4
#	return C2(c, Mv, rv) * x / c * num / den
#
#ddrhopsi_Math_list = [ddrhopsi_Math(el, c=c1, Vv=Vv1, Mv=Mv1) for el in s_list]
#
#
#fig6 = plt.figure(num='ddrhopsi', figsize=(10, 7), dpi=100)
#ax6 = fig6.add_subplot(111)
#ax6.set_xlabel(r'$s$', fontsize=20)
#ax6.set_ylabel(r'$\frac{d^2 \rho}{d \Psi^2}$', fontsize=20)
#ax6.set_xscale('log')
#ax6.set_yscale('log')
#ax6.plot(s_list, ddrhopsi_list, color ='red', linestyle = '-', lw=2.0, label=r'python')
#ax6.plot(s_list, ddrhopsi_interp_list, color ='orange', linestyle = '--', lw=2.0, label=r'interpolate')
#ax6.plot(s_list, ddrhopsi_Math_list, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
#ax6.grid(False)
#ax6.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
#ax6.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
#ax6.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
#ax6.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
#ax6.legend(loc='lower left', prop={'size': 18})
#fig6.tight_layout()
#fig6.show()
#
#
##raw_input()
##sys.exit()
#
#
#print "\nIntegration in s..."
#
#def ints(s, eps) :
#	return 1.0 / (np.sqrt(8.0) * np.pi**2) * dPsi(s, c=c1, Vv=Vv1) * ddrhopsi_interp(s) / np.sqrt(eps - Psi(s, c=c1, Vv=Vv1))
#
#
#eps_list = np.logspace(start=-6, stop=np.log10(17680.0),num=1e4)
#s_eps_list = [sPsi(el, c=c1, Vv=Vv1, approx=False) for el in eps_list]
#sol_s_list = [integrate.quad(lambda s: ints(s, el), 1.0e20, el, epsabs=1.0e-13, epsrel=1.0e-13, limit=100) for el in s_eps_list]
#feps_s_list = [el[0] for el in sol_s_list]
#feps_s_err_list_p = [el[0]+el[1] for el in sol_s_list]
#feps_s_err_list_m = [el[0]-el[1] for el in sol_s_list]
#
#
#fig7 = plt.figure(num='feps_s', figsize=(10, 7), dpi=100)
#ax7 = fig7.add_subplot(111)
#ax7.set_xlabel(r'$\epsilon$', fontsize=20)
#ax7.set_ylabel(r'$f (\epsilon)$', fontsize=20)
#ax7.set_xscale('linear')
#ax7.set_yscale('log')
#ax7.plot(eps_list, feps_s_list, color ='red', linestyle = '-', lw=2.0, label=r'python')
#ax7.fill_between(eps_list, feps_s_err_list_m, feps_s_err_list_p, color ='orange', alpha=0.3)
#ax7.plot(ftab_eps_approx, ftab_feps_approx, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
#ax7.grid(False)
#ax7.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
#ax7.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
#ax7.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
#ax7.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
#ax7.legend(loc='lower left', prop={'size': 18})
#fig7.tight_layout()
#fig7.show()
#
#
#print "\nsol_s_list[0:10] = ", sol_s_list[0:10]
#
#
##raw_input()
##sys.exit()


## check accuracy of rhopsi for approx solution ## --> perfect agreement also with interpolate!
rhopsi_approx_list = [rhopsi(el, c=c1, Vv=Vv1, Mv=Mv1, approx=True) for el in Psi_list] # compared to the Mathematica expression
rhopsi_approx_interp = interpolate.InterpolatedUnivariateSpline(x=Psi_list[::-1], y=rhopsi_approx_list[::-1], k=3)
rhopsi_approx_interp_list = rhopsi_approx_interp(Psi_list)

def rhopsi_Math(psi, c, Vv, Mv=False, rv=False) :
	spsi = sPsi_Math(psi, c, Vv)
	return C2(c, Mv, rv) / (spsi * (1.0 + c * spsi)**2)
	#rv = R200(Mv)
	#return (- 0.04547284088339867 * c**2 * Mv * psi) / (rv**3 * np.log((psi * (- (c / (1.0 + c)) + np.log(1.0 + c))) / (c * Vv**2)) * (Vv - (1.75 * c * Vv**3 * np.log((psi * (- (c / (1.0 + c)) + np.log(1.0 + c))) / (c * Vv**2))) / (psi * (- (c / (1.0 + c)) + np.log(1.0 + c))))**2)

rhopsi_Math_list = [rhopsi_Math(el, c=c1, Vv=Vv1, Mv=Mv1, rv=False) for el in Psi_list]


fig2 = plt.figure(num='rhopsi_approx', figsize=(10, 7), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel(r'$\Psi$', fontsize=20)
ax2.set_ylabel(r'$\rho (\Psi)$', fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.plot(Psi_list, rhopsi_approx_list, color ='red', linestyle = '-', lw=2.0, label=r'approx python')
ax2.plot(Psi_list, rhopsi_approx_interp_list, color ='orange', linestyle = '--', lw=2.0, label=r'interpolate python')
ax2.plot(Psi_list, rhopsi_Math_list, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
ax2.grid(False)
ax2.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax2.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax2.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax2.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax2.legend(loc='lower left', prop={'size': 18})
fig2.tight_layout()
fig2.show()



## check first and second derivatives of rhopsi for approx solution ## --> perfect if s_list is in logspace
rhopsi_approx_p_func = rhopsi_approx_interp.derivative(n=1)
rhopsi_approx_p_func_list = [rhopsi_approx_p_func(el) for el in Psi_list]

def rhopsi_p_Math(psi, c, Vv, Mv=False, rv=False) :
	logpsi = np.log(psi / (c * C1(c, Vv)))
	num = -0.10662224073302788 * psi + (0.5597667638483965 * c * C1(c, Vv) + 0.10662224073302788 * psi) * logpsi - 0.5597667638483965 * c * C1(c, Vv) * logpsi**2
	den = C1(c, Vv) * logpsi**2 * (-0.5714285714285714 * psi + 1.0 * c * C1(c, Vv) * logpsi)**3
	return C2(c, Mv, rv) * psi**2 * num / den
	#rv = R200(Mv)
	#return (- 0.04454482372251298 * c**2 * Mv * psi**2 * (-1.0 * c + (1.0 + c) * np.log(1.0 + c))**2 * (psi * np.log(1.0 + c) * (0.19047619047619047 + 0.19047619047619047 * c + (- 0.19047619047619047 - 0.19047619047619047 * c) * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))) + c * (- 0.19047619047619047 * psi + (0.19047619047619047 * psi + (-1.0 - 1.0 * c) * Vv**2) * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2)) + (1.0 + 1.0 * c) * Vv**2 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**2))) / (rv**3 * Vv**2 * np.log((psi * ((-1.0 * c)/(1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**2 * (0.5714285714285714 * c * psi + (-0.5714285714285714 - 0.5714285714285714 * c) * psi * np.log(1.0 + c) + c * (1.0 + 1.0 * c) * Vv**2 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2)))**3)

rhopsi_p_Math_list = [rhopsi_p_Math(el, c=c1, Vv=Vv1, Mv=Mv1, rv=False) for el in Psi_list]


fig3 = plt.figure(num='rhopsi_approx_p', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$\Psi$', fontsize=20)
ax3.set_ylabel(r'$\frac{d\rho}{d\Psi} (\Psi)$', fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.plot(Psi_list, rhopsi_approx_p_func_list, color ='red', linestyle = '-', lw=2.0, label=r'interpolate python')
ax3.plot(Psi_list, rhopsi_p_Math_list, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
ax3.grid(False)
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='lower left', prop={'size': 18})
fig3.tight_layout()
fig3.show()


rhopsi_approx_pp_func = rhopsi_approx_interp.derivative(n=2)
rhopsi_approx_pp_func_list = [rhopsi_approx_pp_func(el) for el in Psi_list]

def rhopsi_pp_Math(psi, c, Vv, Mv=False, rv=False) :
	logpsi = np.log(psi / (c * C1(c, Vv)))
	num = -0.12185398940917472 * psi**2 + (0.8529779258642234 * c * C1(c, Vv) + 0.06092699470458737 * psi) * psi * logpsi + (-2.239067055393586 * c**2 * C1(c, Vv)**2 - 0.4264889629321117 * c * C1(c, Vv) * psi - 5.551115123125783e-17 * psi**2) * logpsi**2 + c * C1(c, Vv) * (2.798833819241983 * c * C1(c, Vv) - 1.1102230246251565e-16 * psi) * logpsi**3 - 1.119533527696793 * c**2 * C1(c, Vv)**2 * logpsi**4
	den = C1(c, Vv) * logpsi**3 * (0.5714285714285714 * psi - 1.0 * c * C1(c, Vv) * logpsi)**4
	return C2(c, Mv, rv) * psi * num / den
	#rv = R200(Mv)
	#return (- 0.08908964744502595 * c**2 * Mv * psi * (-1.0 * c + (1.0 + c) * np.log(1.0 + c))**2 * ((1.0 + 1.0*c)**2 * psi**2 * np.log(1.0 + c)**2 * (0.10884353741496598 - 0.054421768707483005 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2)) + 1.9471661699494807e-17 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**2) + c * psi * np.log(1.0 + c) * ((- 0.21768707482993194 - 0.21768707482993194 * c) * psi + ((0.108843537414966 + 0.108843537414966 * c) * psi - 0.7619047619047618 * (1.0 + 1.0 * c)**2 * Vv**2) * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2)) + ((- 3.894332339898961e-17 - 3.894332339898961e-17 * c) * psi + 0.380952380952381 * (1.0 + 1.0 * c)**2 * Vv**2) * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**2) + c**2 * (0.10884353741496597 * psi**2 + psi * (- 0.054421768707483 * psi + (0.7619047619047619 + 0.7619047619047619 * c) * Vv**2) * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2)) + (1.9471661699494804e-17 * psi**2 + (- 0.380952380952381 - 0.380952380952381 * c) * psi * Vv**2 + 2.0000000000000004 * (1.0 + 1.0 * c)**2 * Vv**4) * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**2 - 2.5000000000000004 * (1.0 + 1.0 * c)**2 * Vv**4 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**3 + 1.0 * (1.0 + 1.0 * c)**2 * Vv**4 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**4))) / (rv**3 * Vv**2 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2))**3 * (0.5714285714285714 * c * psi + (-0.5714285714285714 - 0.5714285714285714 * c) * psi * np.log(1.0 + c) + c * (1.0 + 1.0 * c) * Vv**2 * np.log((psi * ((-1.0 * c) / (1.0 + c) + np.log(1.0 + c))) / (c * Vv**2)))**4)

rhopsi_pp_Math_list = [rhopsi_pp_Math(el, c=c1, Vv=Vv1, Mv=Mv1, rv=False) for el in Psi_list]


fig4 = plt.figure(num='rhopsi_approx_pp', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$\Psi$', fontsize=20)
ax4.set_ylabel(r'$\frac{d^2\rho}{d\Psi^2} (\Psi)$', fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.plot(Psi_list, rhopsi_approx_pp_func_list, color ='red', linestyle = '-', lw=2.0, label=r'interpolate python')
ax4.plot(Psi_list, rhopsi_pp_Math_list, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
ax4.grid(False)
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='lower left', prop={'size': 18})
fig4.tight_layout()
fig4.show()



#eps_list = np.logspace(start=-6, stop=np.log10(17680.0),num=1e4)
eps_list = np.linspace(start=1.0e-6, stop=17680.0,num=1e4)

print "\nIntegration in psi approx..."

## check integration quad for approx solution ##
def integrand_approx(psi, eps) :
		return 1.0 / (np.sqrt(8.0) * np.pi**2) * rhopsi_approx_pp_func(psi) / np.sqrt(eps - psi) #+ rhopsi_approx_p_func(0) / np.sqrt(eps)

sol_approx_list = [integrate.quad(lambda psi: integrand_approx(psi, el), 0, el, epsabs=1.0e-13, epsrel=1.0e-13, limit=100) for el in eps_list]
feps_approx_list = [el[0] for el in sol_approx_list]
feps_approx_err_list_p = [el[0]+el[1] for el in sol_approx_list]
feps_approx_err_list_m = [el[0]-el[1] for el in sol_approx_list]

print "\nsol_approx_list[0:10] = ", sol_approx_list[0:10]



fig5 = plt.figure(num='feps_approx', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$\epsilon$', fontsize=20)
ax5.set_ylabel(r'$f (\epsilon)$', fontsize=20)
ax5.set_xscale('linear')
ax5.set_yscale('log')
ax5.plot(eps_list, feps_approx_list, color ='red', linestyle = '-', lw=2.0, label=r'approx python')
ax5.fill_between(eps_list, feps_approx_err_list_m, feps_approx_err_list_p, color ='orange', alpha=0.3)
ax5.plot(ftab_eps_approx, ftab_feps_approx, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
ax5.grid(False)
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.legend(loc='lower left', prop={'size': 18})
fig5.tight_layout()
fig5.show()



## find 'exact' solution ##
rhopsi_list = [rhopsi(el, c=c1, Vv=Vv1, Mv=Mv1, approx=False) for el in Psi_list]
rhopsi_interp = interpolate.InterpolatedUnivariateSpline(x=Psi_list[::-1], y=rhopsi_list[::-1], k=3)
rhopsi_pp_func = rhopsi_interp.derivative(n=2)

print "\nIntegration in psi exact..."

def integrand(psi, eps) :
		return 1.0 / (np.sqrt(8.0) * np.pi**2) * rhopsi_pp_func(psi) / np.sqrt(eps - psi) #+ rhopsi_p_func(0) / np.sqrt(eps)

sol_list = [integrate.quad(lambda psi: integrand(psi, el), 0, el, epsabs=1.0e-13, epsrel=1.0e-13, limit=100) for el in eps_list]
feps_list = [el[0] for el in sol_list]
feps_err_list_p = [el[0]+el[1] for el in sol_list]
feps_err_list_m = [el[0]-el[1] for el in sol_list]


print "\nsol_list[0:10] = ", sol_list[0:10]


## change units ##
E_list_unit = [-el/Eunit for el in eps_list]
E_list_unit_r = E_list_unit[::-1]


feps_approx_list_unit = [el/fEunit for el in feps_approx_list]
feps_approx_list_unit_r = feps_approx_list_unit[::-1]
feps_list_unit = [el/fEunit for el in feps_list]
feps_list_unit_r = feps_list_unit[::-1]

ftab_E_unit_approx = [-el/Eunit for el in ftab_eps_approx]
ftab_E_unit_approx_r = ftab_E_unit_approx[::-1]
ftab_feps_unit_approx = [el/fEunit for el in ftab_feps_approx]
ftab_feps_unit_approx_r = ftab_feps_unit_approx[::-1]


## try interpolation before exporting data ##
fE_interp_r = interpolate.InterpolatedUnivariateSpline(x=E_list_unit_r, y=feps_list_unit_r, k=3)
fE_interp_r_list = [fE_interp_r(el) for el in E_list_unit_r]


### export data in .dat file ##
#file_a = open(str_a, "w+")
##file_a.write("#eps\tfeps\n")
#if len(eps_list) == len(feps_list) :
#	for i in range(0, len(eps_list)) :
#		file_a.write("%.8e\t%.8e\n" %(eps_list[i], feps_list[i]))
#
##if len(eps_list) == len(feps_approx_list) :
##	for i in range(0, len(eps_list)) :
##		file_a.write("%.8e\t%.8e\n" %(eps_list[i], feps_approx_list[i]))
#file_a.close()



## Plot ##
fig1 = plt.figure(num='fE', figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r'$E$ [$V_v^2$]', fontsize=20)
ax1.set_ylabel(r'$f (E)$ [$\sqrt{8} M_v / (r_v V_v)^3$]', fontsize=20)
ax1.set_xscale('linear')
ax1.set_yscale('log')
#ax1.set_xlim(-4.25, 0.1)
#ax1.set_ylim(1.0e-9, 1.0e0)
ax1.plot(E_list_unit_r, feps_list_unit_r, color ='red', linestyle = '-', lw=2.0, label=r'exact')
ax1.plot(E_list_unit_r, fE_interp_r_list, color ='green', linestyle = ':', lw=2.0, label=r'interpolate python')
ax1.plot(E_list_unit_r, feps_approx_list_unit_r, color ='orange', linestyle = '--', lw=2.0, label=r'approx python')
ax1.plot(ftab_E_unit_approx_r, ftab_feps_unit_approx_r, color ='blue', linestyle = '-.', lw=2.0, label=r'approx Mathematica')
ax1.grid(False)
ax1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax1.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='all', numticks=15))
ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.legend(loc='lower left', prop={'size': 18})
fig1.tight_layout()
fig1.show()


raw_input()

