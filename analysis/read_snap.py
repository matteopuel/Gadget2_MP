import os.path
import sys

import struct
import numpy as np
from matplotlib import rc

import pygadgetreader as pg


#infile = "../PREFLIGHT/ics_15Mpc_128part_z45_kmax45"
infile = "../out/snp_001"

with open(infile, mode='rb') as file :
	filecontent = file.read()
	file.close()

Ntot = struct.unpack('iiiiii', filecontent[4: 28])
Mtot = struct.unpack('dddddd', filecontent[28: 28+8*6])
N = sum(Ntot)

head = filecontent[0:268]

print "time = ", pg.readheader(infile,'time'), "\n"


PPosGas = [] # they must be strings! (at the end yes, but I think it would be better to work with lists first)
PVelGas = []
PIdGas = []
#PMassGas = []
NendGas = Ntot[0]
for i in range(0, NendGas) :
	PPosGas.append(filecontent[268+12*i:268+12*i+12])
	PVelGas.append(filecontent[276+12*N+12*i:276+12*N+12*i+12])
	PIdGas.append(filecontent[284+24*N+4*i:284+24*N+4*i+4])
	#PMassGas.append(filecontent[292+28*N+4*i:292+28*N+4*i+4])

PPosDM = []
PVelDM = []
PIdDM = []
PMassDM = []
PAngoscDM = []
#PPoscDM = []
NendDM = Ntot[0] + Ntot[1]
for i in range(NendGas, NendDM) :
	PPosDM.append(filecontent[268+12*i:268+12*i+12])
	PVelDM.append(filecontent[276+12*N+12*i:276+12*N+12*i+12])
	PIdDM.append(filecontent[284+24*N+4*i:284+24*N+4*i+4])
	#PMassDM.append(filecontent[292+28*N+4*i:292+28*N+4*i+4])
	PAngoscDM.append(filecontent[292+28*N+4*i:292+28*N+4*i+4]) # 8 instead of 4 becuase of sizeof(double) = 8
	#PPoscDM.append(filecontent[300+28*N +4*(Ntot[1]+Ntot[5]) +4*i:300+28*N +4*(Ntot[1]+Ntot[5]) +4*i+4]) # 8 instead of 4 becuase of sizeof(double) = 8
	
PPosDisk = []
PVelDisk = []
PIdDisk = []
PMassDisk = []
NendDisk = Ntot[0] + Ntot[1] + Ntot[2]
for i in range(NendDM, NendDisk) :
	PPosDisk.append(filecontent[268+12*i:268+12*i+12])
	PVelDisk.append(filecontent[276+12*N+12*i:276+12*N+12*i+12])
	PIdDisk.append(filecontent[284+24*N+4*i:284+24*N+4*i+4])
	#PMassDisk.append(filecontent[292+28*N+4*i:292+28*N+4*i+4])
	
PPosBulge = []
PVelBulge = []
PIdBulge = []
PMassBulge = []
NendBulge = Ntot[0] + Ntot[1] + Ntot[2] + Ntot[3]
for i in range(NendDisk, NendBulge) :
	PPosBulge.append(filecontent[268+12*i:268+12*i+12])
	PVelBulge.append(filecontent[276+12*N+12*i:276+12*N+12*i+12])
	PIdBulge.append(filecontent[284+24*N+4*i:284+24*N+4*i+4])
	#PMassBulge.append(filecontent[292+28*N+4*i:292+28*N+4*i+4])
	
PPosStar = []
PVelStar = []
PIdStar = []
PMassStar = []
NendStar = Ntot[0] + Ntot[1] + Ntot[2] + Ntot[3] + Ntot[4]
for i in range(NendBulge, NendStar) :
	PPosStar.append(filecontent[268+12*i:268+12*i+12])
	PVelStar.append(filecontent[276+12*N+12*i:276+12*N+12*i+12])
	PIdStar.append(filecontent[284+24*N+4*i:284+24*N+4*i+4])
	#PMassStar.append(filecontent[292+28*N+4*i:292+28*N+4*i+4])
	
PPosBndry = []
PVelBndry = []
PIdBndry = []
PMassBndry = []
NendBndry = Ntot[0] + Ntot[1] + Ntot[2] + Ntot[3] + Ntot[4] + Ntot[5]
for i in range(NendStar, NendBndry) :
	PPosBndry.append(filecontent[268+12*i:268+12*i+12])
	PVelBndry.append(filecontent[276+12*N+12*i:276+12*N+12*i+12])
	PIdBndry.append(filecontent[284+24*N+4*i:284+24*N+4*i+4])
	PMassBndry.append(filecontent[292+28*N+4*(i-NendStar):292+28*N+4*(i-NendStar)+4]) # only the Bndry particles change masses because of zoom-in! 


print "struct.unpack('fff', PPosDM[0]) = ", struct.unpack('fff', PPosDM[0])
print "Pos of DM particle 0: ", pg.readsnap(infile, "pos", "dm")[0]
print "Vel of DM particle 0: ", pg.readsnap(infile, "vel", "dm")[0]
print "ID of DM particle 0: ", pg.readsnap(infile, "pid", "dm")[0]
print "mass of DM particle 0: ", pg.readsnap(infile, "mass", "dm")[0]
print pg.readheader(infile, "header")
print "\n"
print "struct.unpack('f', PAngoscDM[0]) = ", struct.unpack('f', PAngoscDM[0])
#print "struct.unpack('f', PPoscDM[0]) = ", struct.unpack('f', PPoscDM[0])
print "struct.unpack('f', PAngoscDM[1]) = ", struct.unpack('f', PAngoscDM[1])
#print "struct.unpack('f', PPoscDM[1]) = ", struct.unpack('f', PPoscDM[1])



print "struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+9*4:28+8*6+24+24+2*4+4*8+9*4+4]) = ", struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+9*4:28+8*6+24+24+2*4+4*8+9*4+4])

