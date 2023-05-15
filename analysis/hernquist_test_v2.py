import numpy as np
from matplotlib import rc
#import pylab as plt
import matplotlib.pyplot as plt
import sys
from itertools import chain
import struct
import pygadgetreader as pg


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


# for NFW matching (...not used)
def rhoH(r, M, a) :
	return M / (2.0 * np.pi) * (a / r) / (r + a)**3

def MHlr(r, M, a) :
	return M * r**2 / (r + a)**2

def rhoNFW(r, rhos, rs) :
	return rhos / (r / rs) / (1.0 + r / rs)**2

def M200(r200) :
	return 4.0 * np.pi / 3.0 * r200**3 * 200.0 * rhocrit 

def R200(m200) :
	r3 = 3.0 * m200 / (4.0 * np.pi * 200.0 * rhocrit)
	return r3**(1.0/3.0)

def aHvsNFW(r200, c) :
	return r200 / (np.sqrt(c**2 / 2.0 / (np.log(1.0 + c) - c / (1.0 + c))) - 1.0)

def MHvsNFW(r200, c) :
	a = aHvsNFW(r200, c)
	m200 = M200(r200)
	return m200 * (r200 + a)**2 / r200**2

def R200_NFW(M, a) :
	cc = 3.0 * M / (4.0 * np.pi * 200.0 * rhocrit)
	return 1.0/3.0 * ( ((3.0 * np.sqrt(3.0 * (4.0 * a**3 * cc + 27.0 * cc**2)) + 2.0 * a**3 + 27.0 * cc) / 2.0)**(1.0/3.0) + 2.0**(1.0/3.0) * a**2 / (3.0 * np.sqrt(3.0 * (4.0 * a**3 * cc + 27.0 * cc**2)) + 2.0 * a**3 + 27.0 * cc)**(1.0/3.0) - 2.0 * a )


# for rejection sampling algorithm
def fE(E, M, a) :
	vg = np.sqrt(G * M / a)
	q = np.sqrt(- a * E / (G * M)) # E < 0 for bound systems
	arg = 3.0 * np.arcsin(q) + q * np.sqrt(1.0 - q**2) * (1.0 - 2.0 * q**2) * (8.0 * q**4 - 8.0 * q**2 - 3.0)
	return M / (8.0 * np.sqrt(2.0) * np.pi**3 * a**3 * vg**3) * arg / (1.0 - q**2)**(2.5)
	#return M / (8.0 * np.sqrt(2.0) * np.pi * a**3 * vg**3) * arg / (1.0 - q**2)**(2.5)

def Phi(r, M, a) :
	return - G * M / (r + a)

def vesc(r, M, a) :
	return np.sqrt(2.0 * G * M / (r + a))


# for spherical symmetry: https://ui.adsabs.harvard.edu/abs/1974A%26A....37..183A/abstract
def spherical(r) :
	phi = 2.0 * np.pi * np.random.uniform(low=0.0, high=1.0, size=None) # same as in code
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)

	cos_theta = 1.0 - 2.0 * np.random.uniform(low=0.0, high=1.0, size=None) # same as in code
	sin_theta = np.sin(np.arccos(cos_theta))

	x = r * sin_theta * cos_phi
	y = r * sin_theta * sin_phi
	z = r * cos_theta

	return [x, y, z]

def getRad(x, y, z) :
	return np.sqrt(np.square(x) + np.square(y) + np.square(z))


# find Xmax based on a cut-off in position space
def Xm(rmax, M, a) :
	xm = rmax**2 / (rmax + a)**2
	print "M(<rmax) = ", xm * M
	print "Xmax = ", xm
	return



## Hernquist and simulation parameters in code units!
##---------# (used for test 2) #---------#
#outfile = "hernquist_test_v2"
#M = 1.0e5 # * 1e10 Msun (total mass -> galaxy cluster)
#a = 1.0e3 # kpc (scale radius)
#Ntot = int(1e6)
#eps = 12.0 # kpc
#dt = 2.5 # Gyr
#boxside = 300000.0 # kpc (= 300 Mpc)
##--> TimeEnd = 2.55503004526 if TimeBegin = 0

##-------# (used for stability) #-------#
#outfile = "hernquist_test2_v2"
#M = 1.0e4 # * 1e10 Msun
#a = 225.0 # kpc
#Ntot = int(128**3)
#eps = 4.4 # kpc
#dt = 10.0 # Gyr
#boxside = 150000.0 # kpc (= 150 Mpc)
##--> TimeEnd = 10.220120181 if TimeBegin = 0

#-------# (used for checks) #-------#
outfile = "hernquist_test3_v2"
M = 3.0 # * 1e10 Msun
a = 10.0 # kpc
Ntot = int(128**3)
eps = 0.3 # kpc
dt = 10.0 # Gyr
boxside = 100000.0 # kpc (= 100 Mpc)
#--> TimeEnd = 10.220120181 if TimeBegin = 0


mp = M / Ntot # * 1e10 Msun --> mp = 0.1
dt *= (Gyr / UnitTime_in_s) # * 0.979 Gyr (code units)

print "dt = ", dt 


## other parameters for the header (come from the parameterfile!) 
tin = 0 # initial time as in TimeBegin (physical)
#z = 1/tin - 1.0 # redshift if tin is the scale factor (when ComovingIntegrationOn = 1)
z = 0 # not relevant
Om0 = 0.31582309
OmL = 0.684176909



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

Xmax = 0.99
M /= Xmax # this is the consequence of considering a cut-off in the radius (divide by Xmax!)


# we generate reflected initial conditions in order to center the halo as in https://iopscience.iop.org/article/10.1086/317149/pdf
if Ntot % 2 == 0 :
	Ntothalf = int(Ntot/2)
else :
	sys.exit()


# METHOD 2: more efficient but less stable
for i in range(0, Ntothalf) :
	## positions
	X = np.random.uniform(low=0, high=1.0, size=None) * Xmax # single value (0 is included, but not 1) # multiplied by Xmax because otherwise domain is too large!
	r = a * np.sqrt(X) / (1.0 - np.sqrt(X))

	xyz = spherical(r)

	x_com += (xyz[0] * mp / Xmax / M)
	y_com += (xyz[1] * mp / Xmax / M)
	z_com += (xyz[2] * mp / Xmax / M)

	pos_tmp.append(xyz)


	## velocities: rejection sampling algorithm
	Phi_r = Phi(r, M, a)
	fE_Phi = fE(Phi_r, M, a)

	q = 0 # define q = vtot / vesc (note that Phi_r = - 0.5 * vesc**2)
	Y = 1.1 # fE_Phi is max of fE() for a given r since f(E) is monotonically decreases with E
	while (Y * fE_Phi) > fE(Phi_r * (1.0 - q**2), M, a) : # first instance must be True
		q = (np.random.uniform(low=0, high=1.0, size=None))**(1.0/3) # **1/3 because the enclosed volume for a sphere goes as r**3
		Y = np.random.uniform(low=0, high=1.0, size=None)
	vtot = q * vesc(r, M, a) # see http://www.artcompsci.org/kali/vol/plummer/ch04.html and https://ui.adsabs.harvard.edu/abs/1974A%26A....37..183A/abstract

	vxyz = spherical(vtot)

	vx_com += (vxyz[0] * mp / Xmax / M) # Xmax dependence should be cancel out
	vy_com += (vxyz[1] * mp / Xmax / M)
	vz_com += (vxyz[2] * mp / Xmax / M)

	vel_tmp.append(vxyz)

	pid_tot.append(i)


# reflected points
for i in range(0, Ntot - Ntothalf) :
	xyz = [-pos_tmp[i][0], -pos_tmp[i][1], -pos_tmp[i][2]]
	vxyz = [-vel_tmp[i][0], -vel_tmp[i][1], -vel_tmp[i][2]]

	x_com += (xyz[0] * mp / Xmax / M)
	y_com += (xyz[1] * mp / Xmax / M)
	z_com += (xyz[2] * mp / Xmax / M)
	pos_tmp.append(xyz)

	vx_com += (vxyz[0] * mp / Xmax / M) # Xmax dependence should be cancel out
	vy_com += (vxyz[1] * mp / Xmax / M)
	vz_com += (vxyz[2] * mp / Xmax / M)
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


outdir = "test2/" 
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


