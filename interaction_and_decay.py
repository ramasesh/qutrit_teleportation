###
# Define Hamiltonian, decay rates, and Lindbladian of system
###

from importlib import reload
import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

import yaml

with open('qutrit_decay.yaml', 'r') as f:
    decay_parameters = yaml.load(f)
with open('ZZ_eigenergies.yaml', 'r') as f:
    ZZ_eigenergies = yaml.load(f)

qutrit_ordering = [1,0,7,6,5]
pairs = [(qutrit_ordering[i], qutrit_ordering[i+1]) for i in range(len(qutrit_ordering) - 1)]
pair_strings = {pair:'Q{}Q{}'.format(pair[0], pair[1]) for pair in pairs}

gamma1GE = [1./decay_parameters['Q{}'.format(q)]['T1_GE'] for q in qutrit_ordering]
gamma1EF = [1./decay_parameters['Q{}'.format(q)]['T1_EF'] for q in qutrit_ordering]
gammaphiEF = [1./decay_parameters['Q{}'.format(q)]['Tphi_EF'] for q in qutrit_ordering]
gammaphiGE = [1./decay_parameters['Q{}'.format(q)]['Tphi_GE'] for q in qutrit_ordering]
gammaphiFG = [1./decay_parameters['Q{}'.format(q)]['T1_EF'] for q in qutrit_ordering]

alpha_11, alpha_12, alpha_21, alpha_22 = [], [], [], []
for pair in pairs:
    alpha_11.append(ZZ_eigenergies[pair_strings[pair]]['11'])
    alpha_12.append(ZZ_eigenergies[pair_strings[pair]]['12'])
    alpha_21.append(ZZ_eigenergies[pair_strings[pair]]['21'])
    alpha_22.append(ZZ_eigenergies[pair_strings[pair]]['22'])
alpha_11 = 2*np.pi*np.array(alpha_11)
alpha_12 = 2*np.pi*np.array(alpha_12)
alpha_21 = 2*np.pi*np.array(alpha_21)
alpha_22 = 2*np.pi*np.array(alpha_22)

# Parameters for specific qutrits:
###

#Interactions for each neighboring pair of qutrits, as a length 4 list for (10) (07) (76) (65) pairs
#alpha_11 = [2 * np.pi * -0.27935, 2 * np.pi * -.1382, 2 * np.pi * -0.276, 2 * np.pi * -0.26175] # MHz
#alpha_12 = [2 * np.pi * 0.1599, 2 * np.pi * .15827, 2 * np.pi * -0.6313, 2 * np.pi * -0.49503] # MHz
#alpha_21 = [2 * np.pi * -0.52793, 2 * np.pi * -.33507, 2 * np.pi * 0.24327, 2 * np.pi * 0.14497] # Hz
#alpha_22 = [2 * np.pi * -0.742967, 2 * np.pi * -.3418, 2 * np.pi * -0.74777, 2 * np.pi * -0.70843] # MHz

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

Z0 = np.diag([1,-1,-1])/np.sqrt(np.sqrt(3))
Z1 = np.diag([-1,1,-1])/np.sqrt(np.sqrt(3))
Z2 = np.diag([-1,-1,1])/np.sqrt(np.sqrt(3))

