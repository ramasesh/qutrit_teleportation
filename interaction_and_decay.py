###
# Define Hamiltonian, decay rates, and Lindbladian of system
###

from importlib import reload
import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

import embed_functions
reload(embed_functions)
from embed_functions import *

num_q = 2

###
# Parameters for specific qutrits:
###

#Decay rates of all 5 qutrits, as a length-5 list for each decay
#gamma1GE = [1./(40),1./(50)] # 1/s #original values
#gamma1EF = [1./(26),1./(33)] # 1/s
gamma1GE = [1./(50.),1./(50.),1./(50.),1./(50.),1./(50.)]
gamma1EF = [1./(30.),1./(30.),1./(30.),1./(30.),1./(30.)]

gammaphiGE = [1/(60),1/(60),1/(60),1/(60),1/(60)] # 1/s
gammaphiEF = [1/(45),1/(45),1/(45),1/(45),1/(45)] # 1/s
# gammaphiGE = [0,0,0,0,0] # 1/s #set to zero for now
# gammaphiEF = [0,0,0,0,0] # 1/s
gammaphiFG = [0,0,0,0,0] # 1/s

#Interactions for each neighboring pair of qutrits, as a length 4 list for (10) (07) (76) (65) pairs
alpha_11 = [2 * np.pi * -0.27935, 2 * np.pi * -.1382, 2 * np.pi * -0.276, 2 * np.pi * -0.26175] # MHz
alpha_12 = [2 * np.pi * 0.1599, 2 * np.pi * .15827, 2 * np.pi * -0.6313, 2 * np.pi * -0.49503] # MHz
alpha_21 = [2 * np.pi * -0.52793, 2 * np.pi * -.33507, 2 * np.pi * 0.24327, 2 * np.pi * 0.14497] # Hz
alpha_22 = [2 * np.pi * -0.742967, 2 * np.pi * -.3418, 2 * np.pi * -0.74777, 2 * np.pi * -0.70843] # MHz

###
# Lots of "fake" couplings to check code was working as expected:
###
# #Interactions for each neighboring pair of qutrits, as a length 4 list for (10) (07) (76) (65) pairs
# alpha_11 = [2 * np.pi * -0.27935, 0, 2 * np.pi * -0.276, 2 * np.pi * -0.26175] # MHz
# alpha_12 = [2 * np.pi * 0.1599, 0, 2 * np.pi * -0.6313, 2 * np.pi * -0.49503] # MHz
# alpha_21 = [2 * np.pi * -0.52793, 0, 2 * np.pi * 0.24327, 2 * np.pi * 0.14497] # Hz
# alpha_22 = [2 * np.pi * -0.742967, 0, 2 * np.pi * -0.74777, 2 * np.pi * -0.70843] # MHz

# #PRETEND ALPHAS if 07 and 65 were DECOUPLED, and 10, 76 had ACTUAL couplings
# alpha_11 = [2 * np.pi * -0.27935, 0, 2 * np.pi * -0.276, 0] # MHz
# alpha_12 = [2 * np.pi * 0.1599, 0, 2 * np.pi * -0.6313, 0] # MHz
# alpha_21 = [2 * np.pi * -0.52793, 0, 2 * np.pi * 0.24327, 0] # Hz
# alpha_22 = [2 * np.pi * -0.742967, 0, 2 * np.pi * -0.74777, 0] # MHz

# #PRETEND ALPHAS if 07 and 65 were NOT DECOUPLED, and 10, 76 had identical but OPPOSITE coupling
# alpha_11 = [2 * np.pi * -0.27935, 2 * np.pi * -0.1382, 2 * np.pi * 0.27935, 2 * np.pi * -0.26175] # MHz
# alpha_12 = [2 * np.pi * 0.1599, 2 * np.pi * 0.15827, 2 * np.pi * -0.1599, 2 * np.pi * -0.49503] # MHz
# alpha_21 = [2 * np.pi * -0.52793,2 * np.pi * -0.33507, 2 * np.pi * 0.52793, 2 * np.pi * 0.14497] # Hz
# alpha_22 = [2 * np.pi * -0.742967, 2 * np.pi * -0.3418, 2 * np.pi * 0.742967, 2 * np.pi * -0.70843] # MHz

# #PRETEND ALPHAS if 07 and 65 were NOT DECOUPLED, and 10, 76 had IDENTICAL coupling
# alpha_11 = [2 * np.pi * -0.27935, 2 * np.pi * -0.1382, 2 * np.pi *-0.27935, 2 * np.pi * -0.26175] # MHz
# alpha_12 = [2 * np.pi * 0.1599, 2 * np.pi * 0.15827, 2 * np.pi * 0.1599, 2 * np.pi * -0.49503] # MHz
# alpha_21 = [2 * np.pi * -0.52793,2 * np.pi * -0.33507, 2 * np.pi * -0.52793, 2 * np.pi * 0.14497] # Hz
# alpha_22 = [2 * np.pi * -0.742967, 2 * np.pi * -0.3418, 2 * np.pi * -0.742967, 2 * np.pi * -0.70843] # MHz

# #PRETEND ALPHAS if 07 and 65 were DECOUPLED, and 10, 76 had IDENTICAL coupling
# alpha_11 = [2 * np.pi * -0.27935, 0, 2 * np.pi * -0.27935, 0] # MHz
# alpha_12 = [2 * np.pi * 0.1599, 0, 2 * np.pi * 0.1599, 0] # MHz
# alpha_21 = [2 * np.pi * -0.52793,0, 2 * np.pi * -0.52793, 0] # Hz
# alpha_22 = [2 * np.pi * -0.742967, 0, 2 * np.pi * -0.742967, 0] # MHz

# #PRETEND ALPHAS if 07 and 65 were DECOUPLED, and 10, 76 had identical but OPPOSITE coupling
# alpha_11 = [2 * np.pi * -0.27935, 0, 2 * np.pi * 0.27935, 0] # MHz
# alpha_12 = [2 * np.pi * 0.1599, 0, 2 * np.pi * -0.1599, 0] # MHz
# alpha_21 = [2 * np.pi * -0.52793,0, 2 * np.pi * 0.52793, 0] # Hz
# alpha_22 = [2 * np.pi * -0.742967, 0, 2 * np.pi * 0.742967, 0] # MHz
# #Ts = np.asarray([ 0.09777726,  0.01669412,  0.09777726,  0.01669412])*2*np.pi

###
# General operators used for decay
###

#Set up decay operators
lower21 = np.asarray([[0,0,0],[0,0,1],[0,0,0]]) #depolarizing
lower10 = np.asarray([[0,1,0],[0,0,0],[0,0,0]])
Z21 = np.diag([0,-1,1]) #dephasing
Z10 = np.diag([-1,1,0])
Z02 = np.diag([1,0,-1])
#Vinay's dephasing operators
Z10 = np.diag([1,-1,-1])
Z21 = np.diag([-1,1,-1])
Z02 = np.diag([0,0,0])

###
#Old way of doing things:
###
# H = np.diag([0,0,0,   0,alpha_11[0],alpha_12[0],   0,alpha_21[0],alpha_22[0]])
# H_double = -1j*(np.kron(H,np.eye(9)) - np.kron(np.eye(9),H.conj()))

# #2-qutrit Linblad evolution matrix
# G = np.asarray([np.kron(Lm.conj(),Lm) - (1./2.)*np.kron(np.eye(9),np.dot(Lm.conj().T,Lm)) \
#     - (1./2.)*np.kron(np.dot(Lm.conj().T,Lm),np.eye(9)) for Lm in L]).sum(axis=0)