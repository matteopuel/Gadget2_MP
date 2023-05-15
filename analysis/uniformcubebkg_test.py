import numpy as np
from matplotlib import rc
#import pylab as plt
import matplotlib.pyplot as plt
import sys
from itertools import chain
import struct
import pygadgetreader as pg


# IC generation: uniform cube in a uniform background particle, without gravity #
#
#							L
#				a	 _______________
#			 _______|				|		y-axis
#	^	  	|		|				|		 
#	|	 a	|	c  -|- zero	  bkg	| b 	 /
#	|		|_______|				|		/
#					|_______________|	   L
#	x-axis			
#		 		---> z-axis
#			


##-- DATA --##
ScatteringCrossSection = 1.0 # cm^2/g (it comes from parameter file!)

# GADGET-2 code units
UnitLenght_in_cm = 3.085678e21  # = 1 kpc -> a = 1.0 kpc
UnitMass_in_g = 1.989e43  # = 1e10 Msun
UnitVelocity_in_cm_per_s = 1e5  # = 1 km/s -> v0 = 1.0 km/s
UnitTime_in_s = UnitLenght_in_cm / UnitVelocity_in_cm_per_s

# the followings are already interpreted in code units
a = 10.0 # side of the cube
v0 = 10.0 # intial z velocity for the particles in the cube

Nc0 = int(1e5) # initial particles in the cube
nb0sigmapL = 0.1

b = 1.1 * a # height and depth of the bkg cuboid
L = 2.0 * a # lenght of the bkg cuboid
Vb = b**2 * L # volume of bkg cuboid
vb0 = 0.0 # intial z velocity for the particles in the bkg cuboid
nb0 = 2.5e5 / a**3 # initial density in the bkg cuboid
Nb0 = int(nb0 * Vb) # initial particles in the bkg cuboid (independent of a)

Ntot = int(Nc0 + Nb0) # total number of particles in the system

Ltot = L + a
Ttot = Ltot / v0 # in code units of time (= * 0.979 Gyr)
sigmap = nb0sigmapL / (nb0 * L) # scattering cross section

sigma0 = 1.0 # cm^2/g, it is the physical sigma/m  

# all the particle have the same (simulation) mass and they make up type==1 in GADGET2
mp = (sigmap / sigma0) / ScatteringCrossSection * (UnitLenght_in_cm**2 / UnitMass_in_g) # * 1e10 Msun  
# in principle I do not need it. However, in the simulation code sigmap = mp * (sigma/m) and (sigma/m) comes from parameter file


## other parameters for the header (come from the parameterfile!) 
tin = 0 # initial time (physical)
#z = 1/tin - 1.0 # redshift if tin is the scale factor (when ComovingIntegrationOn = 1)
z = 0
boxside = Ltot + 2 * a # boxsize
Om0 = 0.31582309
OmL = 0.684176909
h = 0.6732117


print "v0 = ", v0, " km/s"
print "a = ", a, " kpc\n"
print "b = ", b, " kpc"
print "L = ", L, " kpc\n"
print "Ltot (z-axis) = ", Ltot, " kpc"
print "Tot = ", Ttot, " (* 0.979) Gyr\n"
print "Nc0 = ", Nc0
print "Nb0 = ", Nb0
print "Ntot = ", Ntot
print "\nsigmap = ", sigmap
print "mp = ", mp
print "\nz = ", z


##-- CREATE IC file in GADGET-2 format --##


##-- Header --##
SKIP_header = struct.pack('i', 256) # 256 bytes # filecontent[0:4] = filecontent[260: 264]

npart = struct.pack('6i', 0, Ntot, 0, 0, 0, 0) # int npart[6]; #filecontent[4: 4+4*6]
mass = struct.pack('6d', 0, mp, 0, 0, 0, 0) # double mass[6]; #filecontent[28: 28+8*6]

time = struct.pack('d', tin) # double time;
redshift = struct.pack('d', z) # double redshift;
flag_sfr = struct.pack('i', 0) # int flag_sfr;
flag_feedback = struct.pack('i', 0) # int flag_feedback;
block1 = time + redshift + flag_sfr + flag_feedback # filecontent[28+8*6: 28+8*6+24]

npartTotal = struct.pack('6I', 0, Ntot, 0, 0, 0, 0) # unsigned int npartTotal[6];

flag_cooling = struct.pack('i', 0) # int flag_cooling;
num_files = struct.pack('i', 1) # int num_files;
BoxSize = struct.pack('d', boxside) # double BoxSize;
Omega0 = struct.pack('d', Om0) # double Omega0;
OmegaLambda = struct.pack('d', OmL) # double OmegaLambda;
HubbleParam = struct.pack('d', h) # double HubbleParam; 
# filecontent[28+8*6+24+24: 28+8*6+24+24+2*4+4*8]

flag_stellarage = struct.pack('i', 0) # int flag_stellarage;                 
flag_metals = struct.pack('i', 0) # int flag_metals;                     
npartTotalHighWord = struct.pack('6I', 0, 0, 0, 0, 0, 0) # unsigned int npartTotalHighWord[6];
flag_entropy_instead_u = struct.pack('i', 0) # int flag_entropy_instead_u;
# filecontent[28+8*6+24+24+2*4+4*8: 28+8*6+24+24+2*4+4*8+9*4]


#-- non-standard block (5*4 bytes) --#
flag_stuff = struct.pack('i', 0) # unknown meaning...
flag_potential = struct.pack('i', 0) # int flag_potential;
flag_fh2 = struct.pack('i', 0) # int flag_fh2;
flag_tmax = struct.pack('i', 0) # int flag_tmax;
flag_delaytime = struct.pack('i', 0) # int flag_delaytime;
non_standard_block = flag_stuff + flag_potential + flag_fh2 + flag_tmax + flag_delaytime
# filecontent[28+8*6+24+24+2*4+4*8+9*4: 28+8*6+24+24+2*4+4*8+9*4+5*4]


# 44 left (in reality 40 because there is a SKIP_header later)
fill = struct.pack('c', ' ')
#numfill = int(256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8 - 4 * 9 - 4 * 5) # the last piece (-4*5) is non-standard
numfill = int(256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8 - 4 * 9 - 4 * 5)
for i in range(1, numfill) :
	fill += struct.pack('c', ' ')

block2 = flag_cooling + num_files + BoxSize + Omega0 + OmegaLambda + HubbleParam + flag_stellarage + flag_metals + npartTotalHighWord + flag_entropy_instead_u + non_standard_block + fill # filecontent[28+8*6+24+24:260]


header = SKIP_header + npart + mass + block1 + npartTotal + block2 + SKIP_header


##-- Pos, Vel, Pid --##
# center of the simulation: midpoint between cube and bkg cuboid, at a/2 and b/2 (see above scheme)
pos_c = []
vel_c = []
pid_c = []
for i in range(0, Nc0) :
	rnds_x = np.random.uniform(-a/2, a/2, None) # x-axis random numbers is generated
	rnds_y = np.random.uniform(-a/2, a/2, None) # x-axis random numbers is generated
	rnds_z = np.random.uniform(-a, 0, None) # z-axis random number is generated
	pos_c.append([rnds_x, rnds_y, rnds_z])
	vel_c.append([0.0, 0.0, v0])
	pid_c.append(i)

pos_b = []
vel_b = []
pid_b = []
for i in range(0, Nb0) :
	rnds_x = np.random.uniform(-b/2, b/2, None) # x-axis random numbers is generated
	rnds_y = np.random.uniform(-b/2, b/2, None) # x-axis random numbers is generated
	rnds_z = np.random.uniform(0, L, None) # z-axis random number is generated
	pos_b.append([rnds_x, rnds_y, rnds_z])
	vel_b.append([0.0, 0.0, 0.0])
	pid_b.append(Nc0 + i)

pos_tmp = pos_c + pos_b
vel_tmp = vel_c + vel_b
pid_tot = pid_c + pid_b

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

outdir = "test1/" 
outfile = "uniformcubebkg_test"
outpath = outdir + outfile


with open(outpath, mode='wb') as file :
	file.write(filecontent)
	file.close()


print "\nDone!\n"


##-- Prediction for the number of scattered particles --##
Nexp = Nc0 * (1.0 - np.exp(-nb0sigmapL))
print "\nNexp = ", int(Nexp)
Nexp_correct = nb0 * a**2 * L - a**2 / sigmap * np.log(1.0 + np.exp(- sigmap * Nc0 / a**2) * (np.exp(nb0sigmapL) - 1.0))
print "Nexp_correct = ", int(Nexp_correct)


##-- find the softening lenghts for plot --##
xaxis_plot = np.logspace(start=-3, stop=0, num=1e1, endpoint=True, base=10.0) # kpc
softlenght = [el/(nb0**(1.0/3)) for el in xaxis_plot] # kpc
print "\nxaxis_plot:"
for i in range(0, len(xaxis_plot)) :
	print xaxis_plot[i]
print "\nsoftlenght:"
for i in range(0, len(xaxis_plot)) :
	print softlenght[i]

##-- find the fixed timesteps --##
nominal_timestep = [0.001, 0.01, 0.1]
timestep = [el*(a/v0) for el in nominal_timestep] # in code units of time (= * 0.979 Gyr)
print "\ntimestep:"
for i in range(0, len(nominal_timestep)) :
	print timestep[i]


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


