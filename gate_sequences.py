import numpy as np
from importlib import reload

import interaction_and_decay
from interaction_and_decay import *

###
# Store gates as lists with each entry either tuple of form (string, int) or float
# Tuples (string , int) = (single-qutrit gate label [see fundamental_gates.py], qutrit # [in 0 1 2 3 4 notation] )
# float = evolve with Hamiltonian + decoherence/dephasing for time float
#
# ZZ4 gate near bottom is the actual gate we're using! Change time of gate there
###

#ZZ1 - earliest way, no decoupling
#Something wrong still
Ts = np.asarray([ 0.09777726,  0.01669412,  0.09777726,  0.01669412])*2*np.pi
Cphase1_gatelist = [Ts[0],("flip12",0),Ts[1],("flip12",1),Ts[2],("flip12",0),Ts[3],("flip12",1)]
ZZ1_gatelist = Cphase1_gatelist + [("Hadamardc",0),("Hadamardc",1)] + Cphase1_gatelist

#ZZ2 - full X pulse decoupling
Delta = .59644
Cphase2_p1 = [Delta/3.,("X",0),("X",1),Delta/3.,("X",0),("X",1)]
#Delta = - 2 pi / (2*(alpha_11 + alpha_22) - alpha_12 - alpha_21)
Delta_76 =  - 2*np.pi/(2*(alpha_11[2] + alpha_22[2]) - alpha_12[2] - alpha_21[2])
Delta_avg = (Delta + Delta_76)/2. #extremely close to each
#Relative phase - 2 pi /3 = 2/3*Delta*(alpha_11 + alpha_22) - 1/3*Delta*(alpha_12 + alpha_21) between diagonal and off-diagonal
Cphase2_p2 = [Delta/3.,("flip12",0),("flip12",1)]
Cphase2_sq = [(1,0,2*np.pi/3),(2,0,2*np.pi/3),(1,1,2*np.pi/3),(2,1,2*np.pi/3)]
Cphase2_gatelist = Cphase2_p1 + Cphase2_p2 + Cphase2_p1 + Cphase2_p2 + Cphase2_sq
ZZ2_gatelist = Cphase2_gatelist + [("Hadamardc",0),("Hadamardc",1)] + Cphase2_gatelist

Cphase2_p1 = [Delta/3.,("X",0),("X",1),("X",2),("X",3),Delta/3.,("X",0),("X",1),("X",2),("X",3)]
Cphase2_p2 = [Delta/3.,("flip12",0),("flip12",1),("flip12",2),("flip12",3)]
Cphase2_sq = [(1,0,2*np.pi/3),(2,0,2*np.pi/3),(1,1,2*np.pi/3),(2,1,2*np.pi/3),(1,2,2*np.pi/3),(2,2,2*np.pi/3),(1,3,2*np.pi/3),(2,3,2*np.pi/3)]
Cphase2_double_gatelist = Cphase2_p1 + Cphase2_p2 + Cphase2_p1 + Cphase2_p2 + Cphase2_sq
ZZ2_double_gatelist = [("Hadamard",1),("Hadamard",2)] + Cphase2_double_gatelist + [("Hadamardc",0),("Hadamardc",1),("Hadamard",2),("Hadamard",3)] + Cphase2_double_gatelist + [("Hadamardc",1),("Hadamardc",2)]

#ZZ3 -
T1 = .398
T2 = .136
Cphase3_gatelist = [T1,("flip01",0),("flip12",1),T2,("flip12",0),("flip02",1),T2,("flip01",0),("flip12",1),T1]
ZZ3_gatelist = Cphase3_gatelist + [("Hadamardc",0),("Hadamardc",1)] + Cphase3_gatelist

#ZZ4
Tw = Delta/3.
Cphase4_p1_gatelist = [Tw,("flip01",0),("flip01",1),("flip12",2),("flip12",3),Tw,("flip12",0),("flip12",1),("flip01",2),("flip01",3)]
Cphase4_sq = [(1,0,2*np.pi/3),(2,0,2*np.pi/3),(1,1,2*np.pi/3),(2,1,2*np.pi/3),(1,2,2*np.pi/3),(2,2,2*np.pi/3),(1,3,2*np.pi/3),(2,3,2*np.pi/3)]
DP1 = 0*(2*Delta/3.)*np.array([0, alpha_11[1] + alpha_12[1], alpha_21[1] + alpha_22[1] ])
DP2 = 0*(2*Delta/3.)*np.array([0, alpha_11[1] + alpha_21[1], alpha_12[1] + alpha_22[1] ])
DP4 = (2*Delta/3.)*np.array([0, alpha_11[3] + alpha_21[3], alpha_12[3] + alpha_22[3] ])
DP = [(1,1,DP1[1]),(2,1,DP1[2]),(1,2,DP2[1]),(2,2,DP2[2]),(1,4,DP4[1]),(2,4,DP4[2])]
Cphase4_gatelist = Cphase4_p1_gatelist + Cphase4_p1_gatelist + Cphase4_p1_gatelist + Cphase4_sq + DP
ZZ4_gatelist = [("Hadamard",1),("Hadamard",2)] + Cphase4_gatelist + [("Hadamardc",0),("Hadamard",1),("Hadamardc",2),("Hadamard",3)] + Cphase4_gatelist + [("Hadamardc",1),("Hadamardc",2)]

standard_scrambler_gatelist = [("flip12",1)] + Cphase4_gatelist + [("Hadamard",0),("Hadamardc",1),("Hadamardc",2),("Hadamard",3)] + Cphase4_gatelist + [("flip12",3)]
EF_scrambler_gatelist = [("Hadamard",1),("Hadamard",2)] + Cphase4_gatelist + [("Hadamardc",0),("Hadamard",1),("Hadamard",2),("Hadamardc",3)] + Cphase4_gatelist + [("Hadamardc",0),("Hadamardc",3)]

#ZZ4 offset (Norm's idea) - doesn't seem to improve, makes slightly worse
Cphase4_p1_gatelist = [Tw/2.,("flip01",0),("flip01",1),Tw/2.,("flip12",2),("flip12",3),Tw/2.,("flip12",0),("flip12",1),Tw/2.,("flip01",2),("flip01",3)]
Cphase4_gatelist = Cphase4_p1_gatelist + Cphase4_p1_gatelist + Cphase4_p1_gatelist
ZZ4_offset_gatelist = Cphase4_gatelist + [("Hadamardc",0),("Hadamardc",1)] + Cphase4_gatelist
