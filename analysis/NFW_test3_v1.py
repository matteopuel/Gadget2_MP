import numpy as np
from matplotlib import rc
#import pylab as plt
import matplotlib.pyplot as plt
import sys
from itertools import chain
import struct
import pygadgetreader as pg

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


def C1(c, Vv) :
	return Vv**2 * gc(c)

def Phi(s, c, Vv) :
	# s = (r / rv), but c = rv / rs
	x = c * s
	if x == 0 :
		return -C1(c, Vv) * c
	else :
		return -C1(c, Vv) * c * np.log(1.0 + x) / x

def gc(c) :
	tmp = np.log(1.0 + c) - c / (1.0 + c)
	return 1.0 / tmp

def Ms(s, c, Mv) :
	x = c * s
	return Mv * gc(c) * (np.log(1.0 + x) - x / (1.0 + x))

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
	x = c * s
	return C2(c, Mv, rv) * c / (x * (1.0 + x)**2)


def vesc(s, c, Vv) :
	return np.sqrt(- 2.0 * Phi(s, c, Vv))


# for spherical symmetry: https://ui.adsabs.harvard.edu/abs/1974A%26A....37..183A/abstract
def spherical(r) :
	phi = 2.0 * np.pi * np.random.uniform(low=0.0, high=1.0, size=None) # same as in code
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)

	cos_theta = 1.0 - 2.0 * np.random.uniform(low=0.0, high=1.0, size=None) # same as in code (in this way, cos_theta is uniformly distributed while theta itself is not)
	sin_theta = np.sin(np.arccos(cos_theta))

	x = r * sin_theta * cos_phi
	y = r * sin_theta * sin_phi
	z = r * cos_theta

	return [x, y, z]

def getRad(x, y, z) :
	return np.sqrt(np.square(x) + np.square(y) + np.square(z))




## NFW and simulation parameters in code units!
##---------# (DDO 154 dSph galaxy) #---------#
outfile = "nfw_test3_dsph_v1_try"
#outfile = "nfw_test3_dsph_v1_4"
Mv1 = 2.3 # * 1e10 Msun (total mass -> dSph galaxy)
rv1 = R200(Mv1) # kpc
rhos1 = 1.5e-3 # * 1e10 Msun/kpc^3 (not used)
rs1 = 3.4 # kpc (not used)
Vv1 = 49.0 # km/s
c1 = 12.2
Ntot = int(128**3)
#####Ntot = int(10**4)
eps = 0.3 # kpc
dt = 10.0 # Gyr
boxside = 600000.0 # kpc (= 600 Mpc)
##--> TimeEnd = 10.220120181 if TimeBegin = 0
###fname1 = '../../ftab.dat' #--> use Mathematica result in "NFW.nb"!
fname1 = 'nfw_test3_dsph.dat' #--> use python in "NFW.py", but removing negative values


###---------# (A2537 galaxy cluster) #---------#
#outfile = "nfw_test3_cluster_v1"
#rv1 = 2050.0 # kpc
#Mv1 = M200(rv1) # * 1e10 Msun (total mass -> galaxy cluster)
#rhos1 = 1.3e-4 # * 1e10 Msun/kpc^3
#rs1 = 442 # kpc
#Vv1 = 1660.0 # km/s
#c1 = 4.63
#Ntot = int(128**3)
#eps = 9.3 # kpc
#dt = 10.0 # Gyr
#boxside = 300000.0 # kpc (= 300 Mpc)
###--> TimeEnd = 10.220120181 if TimeBegin = 0
#fname1 = '../../ftab_c.dat' #--> use Mathematica result in "NFW.nb"!



print "M200(%.2f) = %.2f" % (rv1, Mv1)
print "R200(%.2f) = %.2f" % (Mv1, rv1)
print "Phi(s=0, c=%.2f, Vv=%.1f) = %f" % (c1, Vv1, Phi(s=0, c=c1, Vv=Vv1))

## Max radius in units of rs ##
rOrs_max = 1.0e5 #1.0e4 # cut-off in the radius
print "Phi(s=%.2f, c=%.2f, Vv=%.1f) = %f" % (rOrs_max/c1, c1, Vv1, Phi(s=rOrs_max/c1, c=c1, Vv=Vv1)) # 200 rs
Mmax = Mv1 * gc(c1) * (np.log(1.0 + rOrs_max) - rOrs_max / (1.0 + rOrs_max))
print "Mmax (r = %.1f rs) = %f" % (rOrs_max, Mmax)

#####
#Mmax = Mv1

mp = Mmax / Ntot # * 1e10 Msun
dt *= (Gyr / UnitTime_in_s) # * 0.979 Gyr (code units)

print "\ndt = ", dt 


## other parameters for the header (comes from the parameterfile!) 
tin = 0 # initial time as in TimeBegin (physical)
#z = 1/tin - 1.0 # redshift if tin is the scale factor (when ComovingIntegrationOn = 1)
z = 0 # not relevant
Om0 = 0.31582309
OmL = 0.684176909



## for rejection sampling algorithm
eps_list  = np.loadtxt(fname=fname1, delimiter='\t', usecols = 0)
feps_list = np.loadtxt(fname=fname1, delimiter='\t', usecols = 1)
E_list1 = [-el for el in eps_list]
E_list = E_list1[::-1]
fE_list = feps_list[::-1]
fE_interp = interpolate.InterpolatedUnivariateSpline(x=E_list, y=fE_list, k=3)


fE_interp_list = [fE_interp(el) for el in E_list]

fig1 = plt.figure(num='fE', figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r'$E$', fontsize=20)
ax1.set_ylabel(r'$f (E)$', fontsize=20)
ax1.set_xscale('linear')
ax1.set_yscale('log')
ax1.plot(E_list, fE_interp_list, color ='red', linestyle = '-', lw=2.0, label=r'interpolate')
ax1.grid(False)
ax1.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True, width=1, length=9, labelsize=20)
ax1.xaxis.set_tick_params(which='minor', direction='in', bottom=True, top=True, width=1, length=4.5)
ax1.yaxis.set_tick_params(which='major', direction='in', left=True, right=True, width=1, length=9, labelsize=20)
ax1.yaxis.set_tick_params(which='minor', direction='in', left=True, right=True, width=1, length=4.5)
ax1.legend(loc='lower left', prop={'size': 18})
fig1.tight_layout()
fig1.show()



##-- CREATE IC file in GADGET-2 format --##


##-- Header --##
SKIP_header = struct.pack('i', 256) # 256 bytes # filecontent[0:4] = filecontent[260:264]

npart = struct.pack('6i', 0, Ntot, 0, 0, 0, 0) # int npart[6]; #filecontent[4:4+4*6]
mass = struct.pack('6d', 0, mp, 0, 0, 0, 0) # double mass[6]; #filecontent[28:28+8*6]

time = struct.pack('d', tin) # double time;
redshift = struct.pack('d', z) # double redshift;
flag_sfr = struct.pack('i', 0) # int flag_sfr;
flag_feedback = struct.pack('i', 0) # int flag_feedback;
block1 = time + redshift + flag_sfr + flag_feedback # filecontent[28+8*6:28+8*6+24]

npartTotal = struct.pack('6I', 0, Ntot, 0, 0, 0, 0) # unsigned int npartTotal[6];

flag_cooling = struct.pack('i', 0) # int flag_cooling;
num_files = struct.pack('i', 1) # int num_files;
BoxSize = struct.pack('d', boxside) # double BoxSize;
Omega0 = struct.pack('d', Om0) # double Omega0;
OmegaLambda = struct.pack('d', OmL) # double OmegaLambda;
HubbleParam = struct.pack('d', h) # double HubbleParam; 
# filecontent[28+8*6+24+24+2*4+3*8:28+8*6+24+24+2*4+4*8]

flag_stellarage = struct.pack('i', 0) # int flag_stellarage;                 
flag_metals = struct.pack('i', 0) # int flag_metals;                     
npartTotalHighWord = struct.pack('6I', 0, 0, 0, 0, 0, 0) # unsigned int npartTotalHighWord[6];
flag_entropy_instead_u = struct.pack('i', 0) # int flag_entropy_instead_u;
# filecontent[28+8*6+24+24+2*4+4*8:28+8*6+24+24+2*4+4*8+9*4]


#-- non-standard block (5*4 bytes) --#
flag_stuff = struct.pack('i', 0) # unknown meaning...
flag_potential = struct.pack('i', 0) # int flag_potential;
flag_fh2 = struct.pack('i', 0) # int flag_fh2;
flag_tmax = struct.pack('i', 0) # int flag_tmax;
flag_delaytime = struct.pack('i', 0) # int flag_delaytime;
non_standard_block = flag_stuff + flag_potential + flag_fh2 + flag_tmax + flag_delaytime
# filecontent[28+8*6+24+24+2*4+4*8+9*4:28+8*6+24+24+2*4+4*8+9*4+5*4]


# 44 left (in reality 40 because there is a SKIP_header later)
fill = struct.pack('c', ' ')
#numfill = int(256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8 - 4 * 9 - 4 * 5) # the last piece (-4*5) is non-standard
numfill = int(256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8 - 4 * 9 - 4 * 5)
for i in range(1, numfill) :
	fill += struct.pack('c', ' ')

block2 = flag_cooling + num_files + BoxSize + Omega0 + OmegaLambda + HubbleParam + flag_stellarage + flag_metals + npartTotalHighWord + flag_entropy_instead_u + non_standard_block + fill # filecontent[28+8*6+24+24:260]


header = SKIP_header + npart + mass + block1 + npartTotal + block2 + SKIP_header



##-- Pos, Vel, Pid --## (here because we have to determine boxsize based on max position)
pos_tmp = []
vel_tmp = []
pid_tot = []

x_com = y_com = z_com = 0
vx_com = vy_com = vz_com = 0



# we generate reflected initial conditions in order to center the halo as in https://iopscience.iop.org/article/10.1086/317149/pdf
if Ntot % 2 == 0 :
	Ntothalf = int(Ntot/2)
else :
	sys.exit()


# METHOD 1: easiest and more stable
for i in range(0, Ntothalf) :
	## positions
	X = np.random.uniform(low=0, high=Mmax, size=None) # single value (0 is included, but Mmax is not)
	
	f = lambda x: Mv1 * gc(c1) * (np.log(1.0 + x) - x / (1.0 + x)) # x = c * s = r / rs 
	invf = inversefunc(f, accuracy=6)
	r = (rv1 / c1) * invf(X)

	xyz = spherical(r)

	x_com += (xyz[0] * mp / Mmax) # Mv1)
	y_com += (xyz[1] * mp / Mmax) # Mv1)
	z_com += (xyz[2] * mp / Mmax) # Mv1)

	pos_tmp.append(xyz)


	## velocities: rejection sampling algorithm
	s1 = r / rv1
	Phi_r = Phi(s=s1, c=c1, Vv=Vv1)

	fE_Phi = fE_interp(Phi_r)
	vesc_r = vesc(s=s1, c=c1, Vv=Vv1)

	vx = vy = vz = vesc_r
	
	while getRad(vx, vy, vz) >= vesc_r :
		vx_tmp, vy_tmp, vz_tmp = np.random.uniform(low=-vesc_r, high=vesc_r, size=3)
		vtot_tmp = getRad(vx_tmp, vy_tmp, vz_tmp)

		Phi_r_v2 = Phi_r + 0.5 * vtot_tmp**2
		if Phi_r_v2 > 0 :
			continue

		fE_Phi_v2 = fE_interp(Phi_r_v2)

		Y = np.random.uniform(low=0, high=1.0, size=None)

		if (Y * fE_Phi) <= fE_Phi_v2 :
			vx = vx_tmp
			vy = vy_tmp
			vz = vz_tmp

	vx_com += (vx * mp / Mmax) # Mv1)
	vy_com += (vy * mp / Mmax) # Mv1)
	vz_com += (vz * mp / Mmax) # Mv1)
	vel_tmp.append([vx, vy, vz])

	pid_tot.append(i)


# reflected points
for i in range(0, Ntot - Ntothalf) :
	xyz = [-pos_tmp[i][0], -pos_tmp[i][1], -pos_tmp[i][2]]
	vxyz = [-vel_tmp[i][0], -vel_tmp[i][1], -vel_tmp[i][2]]

	x_com += (xyz[0] * mp / Mmax) # Mv1)
	y_com += (xyz[1] * mp / Mmax) # Mv1)
	z_com += (xyz[2] * mp / Mmax) # Mv1)
	pos_tmp.append(xyz)

	vx_com += (vxyz[0] * mp / Mmax) # Mv1)
	vy_com += (vxyz[1] * mp / Mmax) # Mv1)
	vz_com += (vxyz[2] * mp / Mmax) # Mv1)
	vel_tmp.append(vxyz)

	pid_tot.append(Ntothalf + i)


# Set the initial centre-of-mass velocity of the system of N-body particles to zero by an overall boost.
r_com = [x_com, y_com, z_com]
v_com = [vx_com, vy_com, vz_com]

print "r_com = ", r_com
print "v_com = ", v_com


pos_tot = list(chain.from_iterable(pos_tmp)) # len = 3 * Ntot
vel_tot = list(chain.from_iterable(vel_tmp)) # len = 3 * Ntot

# convert them into strings
Pos_tmp = []
Vel_tmp = []
for i in range(0, 3*Ntot) :
	structpack_pos = struct.pack('f', pos_tot[i])
	structpack_vel = struct.pack('f', vel_tot[i])
	Pos_tmp.append(structpack_pos)
	Vel_tmp.append(structpack_vel)

Pid_tmp = []
for i in range(0, Ntot) :
	structpack_pid = struct.pack('i', pid_tot[i])
	Pid_tmp.append(structpack_pid)

Pos = ''.join(Pos_tmp)
Vel = ''.join(Vel_tmp)
Pid = ''.join(Pid_tmp)


##-- Combining everything in the outputfile --## 
SKIP = struct.pack('i', 12*Ntot)
SKIP2 = struct.pack('i', 4*Ntot)

filecontent = header + SKIP + Pos + SKIP + SKIP + Vel + SKIP + SKIP2 + Pid + SKIP2


outdir = "test3/" 
outpath = outdir + outfile


with open(outpath, mode='wb') as file :
	file.write(filecontent)
	file.close()


print "\nDone!\n"


##-- CHECK --##
infile = outpath
print "\n\nNumber of DM partiles: ", pg.readheader(infile, "dmcount")
print "header: ", pg.readheader(infile, "header")
print "Pos of first DM particle: ", pg.readsnap(infile, "pos", "dm")[0]
print "Vel of first DM particle: ", pg.readsnap(infile, "vel", "dm")[0]
print "Pid of first DM particle: ", pg.readsnap(infile, "pid", "dm")[0]
print "Mass of first DM particle: ", pg.readsnap(infile, "mass", "dm")[0]
print "\nPos of last DM particle: ", pg.readsnap(infile, "pos", "dm")[-1]
print "Vel of last DM particle: ", pg.readsnap(infile, "vel", "dm")[-1]
print "Pid of last DM particle: ", pg.readsnap(infile, "pid", "dm")[-1]
print "Mass of last DM particle: ", pg.readsnap(infile, "mass", "dm")[-1]


raw_input()
