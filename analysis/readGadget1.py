import os
import sys
import struct


def readHeader(filename, strname) :
	with open(filename, mode='rb') as file :
		filecontent = file.read()
		file.close()

	headernames = ['npart', 'mass', 'time', 'redshift', 'flag_sfr', 'flag_feedback', 'npartTotal', 'flag_cooling', 'num_files', 'boxsize', 'Om0', 'OmL', 'h', 'flag_stellarage', 'flag_metals', 'npartTotalHighWord', 'flag_entropy_instead_u', 'flag_stuff', 'flag_potential', 'flag_fh2', 'flag_tmax', 'flag_delaytime']

	if strname == headernames[0] : # npart
		tmp = struct.unpack('6i', filecontent[4: 4+6*4])
	elif strname == headernames[1] : # mass
		tmp = struct.unpack('6d', filecontent[28: 28+6*8])

	elif strname == headernames[2] : # time
		tmp = struct.unpack('d', filecontent[28+6*8: 28+6*8+1*8])[0]
	elif strname == headernames[3] : # redshift
		tmp = struct.unpack('d', filecontent[28+6*8+1*8: 28+6*8+2*8])[0]
	elif strname == headernames[4] : # flag_sfr
		tmp = struct.unpack('i', filecontent[28+6*8+2*8: 28+6*8+2*8+1*4])[0]
	elif strname == headernames[5] : # flag_feedback
		tmp = struct.unpack('i', filecontent[28+6*8+2*8+1*4: 28+6*8+2*8+2*4])[0]

	elif strname == headernames[6] : # npartTotal
		tmp = struct.unpack('6I', filecontent[28+6*8+24: 28+6*8+24+24])

	elif strname == headernames[7] : # flag_cooling
		tmp = struct.unpack('i', filecontent[28+6*8+24+24: 28+6*8+24+24+1*4])[0]
	elif strname == headernames[8] : # num_files
		tmp = struct.unpack('i', filecontent[28+6*8+24+24+1*4: 28+6*8+24+24+2*4])[0]
	elif strname == headernames[9] : # boxsize
		tmp = struct.unpack('d', filecontent[28+8*6+24+24+2*4: 28+8*6+24+24+2*4+1*8])[0]
	elif strname == headernames[10] : # Om0
		tmp = struct.unpack('d', filecontent[28+8*6+24+24+2*4+1*8: 28+8*6+24+24+2*4+2*8])[0]
	elif strname == headernames[11] : # OmL
		tmp = struct.unpack('d', filecontent[28+8*6+24+24+2*4+2*8: 28+8*6+24+24+2*4+3*8])[0]
	elif strname == headernames[12] : # h
		tmp = struct.unpack('d', filecontent[28+8*6+24+24+2*4+3*8: 28+8*6+24+24+2*4+4*8])[0]

	elif strname == headernames[13] : # flag_stellarage
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8: 28+8*6+24+24+2*4+4*8+1*4])[0]
	elif strname == headernames[14] : # flag_metals
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+1*4: 28+8*6+24+24+2*4+4*8+2*4])[0]
	elif strname == headernames[15] : # npartTotalHighWord
		tmp = struct.unpack('6I', filecontent[28+8*6+24+24+2*4+4*8+2*4: 28+8*6+24+24+2*4+4*8+8*4])
	elif strname == headernames[16] : # flag_entropy_instead_u
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+8*4: 28+8*6+24+24+2*4+4*8+9*4])[0]

	elif strname == headernames[17] : # flag_stuff
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+9*4: 28+8*6+24+24+2*4+4*8+9*4+1*4])[0]
	elif strname == headernames[18] : # flag_potential
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+9*4+1*4: 28+8*6+24+24+2*4+4*8+9*4+2*4])[0]
	elif strname == headernames[19] : # flag_fh2
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+9*4+2*4: 28+8*6+24+24+2*4+4*8+9*4+3*4])[0]
	elif strname == headernames[20] : # flag_tmax
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+9*4+3*4: 28+8*6+24+24+2*4+4*8+9*4+4*4])[0]
	elif strname == headernames[21] : # flag_delaytime
		tmp = struct.unpack('i', filecontent[28+8*6+24+24+2*4+4*8+9*4+4*4: 28+8*6+24+24+2*4+4*8+9*4+5*4])[0]

	else :
		print "Possible strname = ", headernames
		sys.exit()

	return tmp


def readSnapshot(filename, ptype, strname='full', full=False, mass=False) :
	with open(filename, mode='rb') as file :
		filecontent = file.read()
		file.close()

	typenames = ['gas', 'dm', 'disk', 'bulge', 'star', 'bndry']
	snapnames = ['pos', 'vel', 'pid', 'mass']

	Ntot = struct.unpack('6i', filecontent[4: 4+6*4])
	N = sum(Ntot)
	NendGas = Ntot[0]
	NendDM = Ntot[0] + Ntot[1]
	NendDisk = Ntot[0] + Ntot[1] + Ntot[2]
	NendBulge = Ntot[0] + Ntot[1] + Ntot[2] + Ntot[3]
	NendStar = Ntot[0] + Ntot[1] + Ntot[2] + Ntot[3] + Ntot[4]
	NendBndry = Ntot[0] + Ntot[1] + Ntot[2] + Ntot[3] + Ntot[4] + Ntot[5]

	Mtot = struct.unpack('6d', filecontent[28: 28+6*8])
	PPos = []
	PVel = []
	PId = []
	PMass = []

	if ptype == typenames[0] :
		for i in range(0, NendGas) :
			PPos.append(struct.unpack('3f', filecontent[268+12*i: 268+12*i+12]))
			PVel.append(struct.unpack('3f', filecontent[276+12*N+12*i: 276+12*N+12*i+12]))
			PId.append(struct.unpack('i', filecontent[284+24*N+4*i: 284+24*N+4*i+4])[0])
			if mass == True :
				PMass.append(struct.unpack('f', filecontent[292+28*N+4*i: 292+28*N+4*i+4])[0])
			else :
				PMass.append(Mtot[0])

	elif ptype == typenames[1] :
		for i in range(NendGas, NendDM) :
			PPos.append(struct.unpack('3f', filecontent[268+12*i: 268+12*i+12]))
			PVel.append(struct.unpack('3f', filecontent[276+12*N+12*i: 276+12*N+12*i+12]))
			PId.append(struct.unpack('i', filecontent[284+24*N+4*i: 284+24*N+4*i+4])[0])
			#PAngoscDM.append(struct.unpack('f', filecontent[292+28*N+4*i: 292+28*N+4*i+4])[0]) # 8 instead of 4 becuase of sizeof(double) = 8
			if mass == True :
				PMass.append(struct.unpack('f', filecontent[292+28*N+4*i: 292+28*N+4*i+4])[0])
			else :
				PMass.append(Mtot[1])

	elif ptype == typenames[2] :
		for i in range(NendDM, NendDisk) :
			PPos.append(struct.unpack('3f', filecontent[268+12*i: 268+12*i+12]))
			PVel.append(struct.unpack('3f', filecontent[276+12*N+12*i: 276+12*N+12*i+12]))
			PId.append(struct.unpack('i', filecontent[284+24*N+4*i: 284+24*N+4*i+4])[0])
			if mass == True :
				PMass.append(struct.unpack('f', filecontent[292+28*N+4*i: 292+28*N+4*i+4])[0])
			else :
				PMass.append(Mtot[2])

	elif ptype == typenames[3] :
		for i in range(NendDisk, NendBulge) :
			PPos.append(struct.unpack('3f', filecontent[268+12*i: 268+12*i+12]))
			PVel.append(struct.unpack('3f', filecontent[276+12*N+12*i: 276+12*N+12*i+12]))
			PId.append(struct.unpack('i', filecontent[284+24*N+4*i: 284+24*N+4*i+4])[0])
			if mass == True :
				PMass.append(struct.unpack('f', filecontent[292+28*N+4*i: 292+28*N+4*i+4])[0])
			else :
				PMass.append(Mtot[3])

	elif ptype == typenames[4] :
		for i in range(NendBulge, NendStar) :
			PPos.append(struct.unpack('3f', filecontent[268+12*i: 268+12*i+12]))
			PVel.append(struct.unpack('3f', filecontent[276+12*N+12*i: 276+12*N+12*i+12]))
			PId.append(struct.unpack('i', filecontent[284+24*N+4*i: 284+24*N+4*i+4])[0])
			if mass == True :
				PMass.append(struct.unpack('f', filecontent[292+28*N+4*i: 292+28*N+4*i+4])[0])
			else :
				PMass.append(Mtot[4])

	elif ptype == typenames[5] :
		for i in range(NendStar, NendBndry) :
			PPos.append(struct.unpack('3f', filecontent[268+12*i: 268+12*i+12]))
			PVel.append(struct.unpack('3f', filecontent[276+12*N+12*i: 276+12*N+12*i+12]))
			PId.append(struct.unpack('i', filecontent[284+24*N+4*i: 284+24*N+4*i+4])[0])
			if mass == True :
				# subtacting NendStar subtracts N_DM, which should have equal mass and so they do not enter the mass block
				PMass.append(struct.unpack('f', filecontent[292+28*N+4*(i-NendStar): 292+28*N+4*(i-NendStar)+4])[0]) # only if the Bndry particles are the only type to change masses! (if other types has different masses, this need to be changed!!!) 
			else :
				PMass.append(Mtot[5])

	else :
		print "Possible ptype = ", typenames
		sys.exit()


	if full == False :
		if strname == snapnames[0] :
			return PPos
		elif strname == snapnames[1] :
			return PVel
		elif strname == snapnames[2] :
			return PId
		elif strname == snapnames[3] :
			return PMass
		else :
			print "Possible strname = ", snapnames
			sys.exit()
	else :
		return [PPos, PVel, PId, PMass]

