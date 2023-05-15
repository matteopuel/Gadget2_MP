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

rc('font', family='serif', size=20)
rc('legend', fontsize=20)
#plt.style.use('classic')


# GADGET-2 code units
UnitLength_in_cm = 3.085678e21  # = 1 kpc -> a = 1.0 kpc
UnitMass_in_g = 1.989e43  # = 1e10 Msun
UnitVelocity_in_cm_per_s = 1e5  # = 1 km/s -> v0 = 1.0 km/s
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s



##-- Import data for first plot --##
hnb1_3 = np.loadtxt(fname='test1/fig_test1.txt', delimiter='\t', usecols = 0) # x-axis
h = np.loadtxt(fname='test1/fig_test1.txt', delimiter='\t', usecols = 1) #softening length
Nexp = np.loadtxt(fname='test1/fig_test1.txt', delimiter='\t', usecols = 2)
Nsim_01 = np.loadtxt(fname='test1/fig_test1.txt', delimiter='\t', usecols = 3)
Nsim_001 = np.loadtxt(fname='test1/fig_test1.txt', delimiter='\t', usecols = 4)
Nsim_0001 = np.loadtxt(fname='test1/fig_test1.txt', delimiter='\t', usecols = 5)

NoNexp_01 = [Nsim_01[i]/Nexp[i] for i in range(0, len(Nexp))]
NoNexp_001 = [Nsim_001[i]/Nexp[i] for i in range(0, len(Nexp))]
NoNexp_0001 = [Nsim_0001[i]/Nexp[i] for i in range(0, len(Nexp))]

# assumption: Nsim is Poisson-distributed -> std = sqrt(Nsim)
dNoNexp_01 = [np.sqrt(Nsim_01[i])/Nexp[i] for i in range(0, len(Nexp))]
dNoNexp_001 = [np.sqrt(Nsim_001[i])/Nexp[i] for i in range(0, len(Nexp))]
dNoNexp_0001 = [np.sqrt(Nsim_0001[i])/Nexp[i] for i in range(0, len(Nexp))]

# for line fit
def Cubic(t, a) :
	return (a * t)**3

hnb1_3_01_sub = [hnb1_3[2], hnb1_3[3], hnb1_3[4], hnb1_3[5]]
NoNexp_01_sub = [NoNexp_01[2], NoNexp_01[3], NoNexp_01[4], NoNexp_01[5]]
coeffCubic_01 = curve_fit(lambda t,a: Cubic(t,a), hnb1_3_01_sub, NoNexp_01_sub)
#print "\ncoeffCubic_01 = ", coeffCubic_01[0][0]
fit_01 = [Cubic(el, coeffCubic_01[0][0]) for el in hnb1_3]

hnb1_3_001_sub = [hnb1_3[1], hnb1_3[2], hnb1_3[3], hnb1_3[4]]
NoNexp_001_sub = [NoNexp_001[1], NoNexp_001[2], NoNexp_001[3], NoNexp_001[4]]
coeffCubic_001 = curve_fit(lambda t,a: Cubic(t,a), hnb1_3_001_sub, NoNexp_001_sub)
#print "\ncoeffCubic_001 = ", coeffCubic_001[0][0]
fit_001 = [Cubic(el, coeffCubic_001[0][0]) for el in hnb1_3]

hnb1_3_0001_sub = [hnb1_3[0], hnb1_3[1], hnb1_3[2], hnb1_3[3]]
NoNexp_0001_sub = [NoNexp_0001[0], NoNexp_0001[1], NoNexp_0001[2], NoNexp_0001[3]]
coeffCubic_0001 = curve_fit(lambda t,a: Cubic(t,a), hnb1_3_0001_sub, NoNexp_0001_sub)
#print "\ncoeffCubic_0001 = ", coeffCubic_0001[0][0]
fit_0001 = [Cubic(el, coeffCubic_0001[0][0]) for el in hnb1_3]



##-- Set the file --##
infile = "test1/timestep0001/pt10/out/snp_007"
#infile = "test1/timestep01/pt10/out/snp_007"

Ntot = readGadget1.readHeader(filename=infile, strname='npart')
Mtot = readGadget1.readHeader(filename=infile, strname='mass')
time = readGadget1.readHeader(filename=infile, strname='time')

print "Ntot = ", Ntot
print "Mtot = ", Mtot
print "time = ", time, "\n"


##-- Read the particle properties --##
PPosDM, PVelDM, PIdDM, _ = readGadget1.readSnapshot(filename=infile, ptype='dm', strname='full', full=True, mass=False)


##-- Plot the velocities of only scattered particles --##
v0 = 10.0 # from uniformcubebkg_test.py
Nc0 = int(1e5) # initial particles in the cube (from uniformcubebkg_test.py)

PVelDM_scattered_c = [] # scattered particles in the cube
PVelDM_scattered_b = [] # scattered particles in the bkg cuboid
for i in range(0, len(PIdDM)) :
	if PIdDM[i] < Nc0 : # cube
		if (PVelDM[i][0] != 0 or PVelDM[i][1] != 0) :
			PVelDM_scattered_c.append(PVelDM[i])
	else : # bkg
		if (PVelDM[i][0] != 0 or PVelDM[i][1] != 0 or PVelDM[i][2] != 0) :
			PVelDM_scattered_b.append(PVelDM[i])


print "scattered cube = ", len(PVelDM_scattered_c)
print "scattered bkg = ", len(PVelDM_scattered_b)


vel_c_x = [PVelDM_scattered_c[i][0] for i in range(0, len(PVelDM_scattered_c))]
vel_c_y = [PVelDM_scattered_c[i][1] for i in range(0, len(PVelDM_scattered_c))]
vel_c_z = [PVelDM_scattered_c[i][2] for i in range(0, len(PVelDM_scattered_c))]
vel_c = [np.sqrt(vel_c_x[i]**2 + vel_c_y[i]**2 + vel_c_z[i]**2) for i in range(0, len(PVelDM_scattered_c))]
vov0_c = [el/v0 for el in vel_c]

theta_c = [np.arccos(vel_c_z[i]/vel_c[i]) for i in range(0, len(PVelDM_scattered_c))]
phi_c = []
for i in range(0, len(PVelDM_scattered_c)) :
	phi_c_i = np.arccos(vel_c_x[i]/(vel_c[i]*np.sin(theta_c[i])))
	if vel_c_y[i] >= 0 :
		phi_c.append(phi_c_i)
	else :
		phi_c.append(np.pi + phi_c_i)

vel_b_x = [PVelDM_scattered_b[i][0] for i in range(0, len(PVelDM_scattered_b))]
vel_b_y = [PVelDM_scattered_b[i][1] for i in range(0, len(PVelDM_scattered_b))]
vel_b_z = [PVelDM_scattered_b[i][2] for i in range(0, len(PVelDM_scattered_b))]
vel_b = [np.sqrt(vel_b_x[i]**2 + vel_b_y[i]**2 + vel_b_z[i]**2) for i in range(0, len(PVelDM_scattered_b))]
vov0_b = [el/v0 for el in vel_b]

theta_b = [np.arccos(vel_b_z[i]/vel_b[i]) for i in range(0, len(PVelDM_scattered_b))]
phi_b = []
for i in range(0, len(PVelDM_scattered_b)) :
	phi_b_i = np.arccos(vel_b_x[i]/(vel_b[i]*np.sin(theta_b[i])))
	if vel_b_y[i] >= 0 :
		phi_b.append(phi_b_i)
	else :
		phi_b.append(np.pi + phi_b_i)


# expected distributions:
numbins = 25

v_list = np.linspace(start=0, stop=v0, num=100)
fv = [2.0 * el for el in v_list]
fv_m = [el - np.sqrt(el/numbins)/2 for el in fv]
fv_p = [el + np.sqrt(el/numbins)/2 for el in fv]

theta_list = np.linspace(start=0, stop=np.pi/2, num=100)
ftheta = [np.sin(2.0 * el) for el in theta_list]
ftheta_m = [el - np.sqrt(np.pi/2 * el/numbins/2)/2 for el in ftheta] # I multiplied by sqrt(1/2) to reduce the error (it was too large otherwise)
ftheta_p = [el + np.sqrt(np.pi/2 * el/numbins/2)/2 for el in ftheta] # I multiplied by sqrt(1/2) to reduce the error (it was too large otherwise)

phi_list = np.linspace(start=0, stop=2.0*np.pi, num=100)
fphi = [1.0 / (2.0 * np.pi) for el in phi_list]
fphi_m = [el - np.sqrt(el/numbins/4)/2 for el in fphi] # I multiplied by 1/2 to reduce the error (it was too large otherwise)
fphi_p = [el + np.sqrt(el/numbins/4)/2 for el in fphi] # I multiplied by 1/2 to reduce the error (it was too large otherwise)


##-- Plot first figure --##
fig1 = plt.figure(num='NoNexp', figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r'$h (\rho_b / m_p)^{1/3}$', fontsize=20)
ax1.set_ylabel(r'$N / N_{\rm exp}$', fontsize=20)
ax1.set_xscale('linear')
ax1.set_yscale('linear')
ax1.set_xlim(0, 1.2)
ax1.set_ylim(0, 1.2)
ax1.errorbar(hnb1_3, NoNexp_01, xerr=0, yerr=dNoNexp_01, c='red', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$\Delta t = 0.1$')
ax1.errorbar(hnb1_3, NoNexp_001, xerr=0, yerr=dNoNexp_001, c='green', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$\Delta t = 0.01$')
ax1.errorbar(hnb1_3, NoNexp_0001, xerr=0, yerr=dNoNexp_0001, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$\Delta t = 0.001$')
ax1.axhline(1.0, xmin=0, xmax=1.4, linestyle='--', lw=1.5, color='black')
ax1.grid(False)
ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
ax1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
handles1, labels1 = ax1.get_legend_handles_labels()
handles1 = [el[0] for el in handles1]
ax1.legend(handles1, labels1, loc='lower right', prop={'size': 18}, numpoints=1)
fig1.tight_layout()
fig1.show()
#fig1.savefig('test1/figs/NoNexp.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


fig2 = plt.figure(num='NoNexp1', figsize=(10, 7), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel(r'$h (\rho_b / m_p)^{1/3}$', fontsize=20)
ax2.set_ylabel(r'$N / N_{\rm exp}$', fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1.0e-3, 1.0)
ax2.set_ylim(1.0e-4, 4.0)
ax2.errorbar(hnb1_3, NoNexp_01, xerr=0, yerr=dNoNexp_01, c='red', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$\Delta t = 0.1$')
ax2.errorbar(hnb1_3, NoNexp_001, xerr=0, yerr=dNoNexp_001, c='green', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$\Delta t = 0.01$')
ax2.errorbar(hnb1_3, NoNexp_0001, xerr=0, yerr=dNoNexp_0001, c='blue', marker='o', mec='black', alpha=1.0, linestyle='None', label=r'$\Delta t = 0.001$')
ax2.axhline(1.0, xmin=0, xmax=1.4, linestyle='--', lw=1.5, color='black')
ax2.plot(hnb1_3, fit_01, color ='red', linestyle = '-', lw=1.5)
ax2.plot(hnb1_3, fit_001, color ='green', linestyle = '-', lw=1.5)
ax2.plot(hnb1_3, fit_0001, color ='blue', linestyle = '-', lw=1.5)
ax2.grid(False)
ax2.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax2.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax2.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax2.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
handles2, labels2 = ax2.get_legend_handles_labels()
handles2 = [el[0] for el in handles2]
ax2.legend(handles2, labels2, loc='lower right', prop={'size': 18}, numpoints=1)
fig2.tight_layout()
fig2.show()
#fig2.savefig('test1/figs/NoNexp1.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



##-- Plot second figure --##
fig0 = plt.figure(num='vel_c', figsize=(10, 7), dpi=100)
ax0 = fig0.add_subplot(111)
ax0.set_xlabel(r'$v$ [km/s] ', fontsize=20)
ax0.set_ylabel(r'Normalized Entries', fontsize=20)
ax0.set_xscale('linear')
ax0.set_yscale('linear')
ax0.set_xlim(-5.0, 10.0)
n1, bins1, _ = ax0.hist(vel_c_x, bins=numbins, range=None, histtype='bar', density=True, edgecolor='blue', color='blue', alpha=0.5, label=r'$v_x$')
n2, bins2, _ = ax0.hist(vel_c_y, bins=numbins, range=None, histtype='bar', density=True, edgecolor='red', color='red', alpha=0.5, label=r'$v_y$')
n3, bins3, _ = ax0.hist(vel_c_z, bins=numbins, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'$v_z$')
ax0.grid(False)
ax0.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax0.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
ax0.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax0.ticklabel_format(axis='y', style='scientific')
ax0.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax0.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax0.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax0.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax0.legend(loc='upper right', prop={'size': 18})
fig0.tight_layout()
fig0.show()


fig3 = plt.figure(num='fv', figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_xlabel(r'$v/v_0$', fontsize=22)
ax3.set_ylabel(r'$f(v)$', fontsize=22)
ax3.set_xscale('linear')
ax3.set_yscale('linear')
ax3.set_xlim(0.001, 1.0)
ax3.set_ylim(0, 2.0)
n1, bins1, _ = ax3.hist(vov0_c, bins=numbins, range=None, histtype='bar', density=True, edgecolor='blue', color='blue', alpha=0.5, label=r'cube')
n2, bins2, _ = ax3.hist(vov0_b, bins=numbins, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'background')
ax3.plot(v_list, fv, color ='red', linestyle = '--', lw=2.5, label=r'analytical')
ax3.fill_between(v_list, fv_m, fv_p, color ='red', alpha=0.2)
ax3.grid(False)
ax3.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax3.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
ax3.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax3.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
ax3.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=22)
ax3.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax3.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=22)
ax3.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax3.legend(loc='upper left', prop={'size': 20})
fig3.tight_layout()
fig3.show()
#fig3.savefig('test1/figs/fv.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


def format_func(value, tick_number):
    # find number of multiples of pi/4
    N = int(np.round(4 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/4$"
    elif N == 2:
        return r"$\pi/2$"
    elif N == 4:
        return r"$\pi$"
    elif N == 6:
        return r"$3\pi/2$"
    elif N == 8:
        return r"$2\pi$"


fig4 = plt.figure(num='ftheta', figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_xlabel(r'$\theta$', fontsize=22)
ax4.set_ylabel(r'$f(\theta)$', fontsize=22)
ax4.set_xscale('linear')
ax4.set_yscale('linear')
ax4.set_xlim(0.001, np.pi/2)
ax4.set_ylim(0, 1.25)
n1, bins1, _ = ax4.hist(theta_c, bins=numbins, range=None, histtype='bar', density=True, edgecolor='blue', color='blue', alpha=0.5, label=r'cube')
n2, bins2, _ = ax4.hist(theta_b, bins=numbins, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'background')
ax4.plot(theta_list, ftheta, color ='red', linestyle = '--', lw=2.5, label=r'analytical')
ax4.fill_between(theta_list, ftheta_m, ftheta_p, color ='red', alpha=0.2)
ax4.grid(False)
ax4.xaxis.set_major_locator(mpl.ticker.MultipleLocator(np.pi/4))
ax4.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(np.pi/8))
ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax4.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax4.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
ax4.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=22)
ax4.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax4.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=22)
ax4.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax4.legend(loc='upper left', prop={'size': 20})
fig4.tight_layout()
fig4.show()
#fig4.savefig('test1/figs/ftheta.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)


fig5 = plt.figure(num='fphi', figsize=(10, 7), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'$\phi$', fontsize=22)
ax5.set_ylabel(r'$f(\phi)$', fontsize=22)
ax5.set_xscale('linear')
ax5.set_yscale('linear')
ax5.set_xlim(0.01, 2.0*np.pi)
ax5.set_ylim(0, 0.275)
n1, bins1, _ = ax5.hist(phi_c, bins=numbins, range=None, histtype='bar', density=True, edgecolor='blue', color='blue', alpha=0.5, label=r'cube')
n2, bins2, _ = ax5.hist(phi_b, bins=numbins, range=None, histtype='bar', density=True, edgecolor='green', color='green', alpha=0.5, label=r'background')
ax5.plot(phi_list, fphi, color ='red', linestyle = '--', lw=2.5, label=r'analytical')
ax5.fill_between(phi_list, fphi_m, fphi_p, color ='red', alpha=0.2)
ax5.grid(False)
ax5.xaxis.set_major_locator(mpl.ticker.MultipleLocator(np.pi/2))
ax5.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(np.pi/4))
ax5.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax5.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
ax5.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.025))
ax5.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=22)
ax5.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax5.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=22)
ax5.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax5.legend(loc='upper left', prop={'size': 20})
fig5.tight_layout()
fig5.show()
#fig5.savefig('test1/figs/fphi.pdf', dpi=200, bbox_inches='tight', pad_inches=0.08)



raw_input()
