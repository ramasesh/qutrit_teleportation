import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import math
from importlib import reload

import interaction_and_decay as iad

import fundamental_gates
reload(fundamental_gates)
from fundamental_gates import *

import embed_functions
reload(embed_functions)
from embed_functions import *

import gate_sequences
reload(gate_sequences)
from gate_sequences import *

###
# Combine parameters and operators to form Hamiltonian and Lindbladian for the system
###

#1-qutrit Lindblad operators
#With dephasing
# L_1q = [[ np.sqrt(iad.gamma1GE[i])*iad.lower10, np.sqrt(iad.gamma1EF[i])*iad.lower21, \
#         np.sqrt(iad.gammaphiGE[i])*Z10, np.sqrt(iad.gammaphiEF[i])*Z21, np.sqrt(iad.gammaphiFG[i])*Z02] for i in range(num_q)]
#Without dephasing
L_1q = [[ np.sqrt(iad.gamma1GE[i])*iad.lower10, np.sqrt(iad.gamma1EF[i])*iad.lower21] for i in range(num_q)]
#2-qutrit Lindblad operators
L = [np.kron(l1,np.eye(3)) for l1 in L_1q[0]] + [np.kron(np.eye(3),l1) for l1 in L_1q[1]]

labelToNum_dict = { 1:0, 0:1, 7:2, 6:3, 5:4 }
numToLabel_dict = { 0:1, 1:0, 2:7, 3:6, 4:5 }


def depol_channel(rho,p,i):
    """
    Implements fully depolarizing channel on qutrit i (use 01234 labelling) with probability p on full density matrix rho
    """
    # rho -> (1-p)*rho + p*1/d = (1-p)*rho + p*sum( P . rho . P^dagger )/d^2
    num_q = round(math.log(np.shape(rho)[0],3)/2) #determine number of qutrits from size of rho
    #Define qutrit pauli matrices
    Xs = [np.eye(3), X, X.T]
    Zs = [np.eye(3), Z, Z.conj()]
    paulis = [Xop.dot(Zop) for Xop in Xs for Zop in Zs]
    #embed paulis onto correct qutrits
    paulis_emb = [embedGate(pauli,i,num_q) for pauli in paulis]
    rho_f = rho*(1.-p)
    for P in paulis_emb:
        rho_f += p*P.dot(rho)/3**2
    return rho_f
    # #Checking that two applications of depol channel correctly gives anticipated fidelity (small 1/3**2 correction from probability of full depolarized being in correct EPR state)
    # p = 1 - np.sqrt(.92)
    # print(p)
    # rho = depol_channel(rho_EPR.reshape((81,1)),p,0).todense().reshape((9,9))
    # rho = depol_channel(rho.reshape((81,1)),p,1).reshape((9,9))
    # np.trace(rho.dot(rho_EPR.todense()))

def getHandG(q_list):
    """
    Enter q_list with Vinay/Machiel index 10765
    """
    q_list_num = [labelToNum_dict[q] for q in q_list]
    pair_list = []
    #Get list of neighboring pairs that are contained in q_list
    for q1 in q_list_num:
        for q2 in q_list_num:
            if q1 == q2 + 1:
                pair_list.append(q2)
    #Get list of 2-qutrit Hamiltonians, and their Lindbladian doubles
    H_2 = [np.diag([0,0,0,   0,iad.alpha_11[i],iad.alpha_12[i],   0,iad.alpha_21[i],iad.alpha_22[i]]) for i in pair_list] #12 gain phase
    H_full = 0.*sps.eye(3**(2*len(q_list)))
    for j in range(len(pair_list)):
        H_full = H_full + embed2Ham(H_2[j],pair_list[j],len(q_list))
        
    #1-qutrit Lindblad operators
    L_1q = [[ np.sqrt(iad.gamma1GE[i])*iad.lower10, np.sqrt(iad.gamma1EF[i])*iad.lower21, \
        np.sqrt(iad.gammaphiGE[i])*iad.Z10, np.sqrt(iad.gammaphiEF[i])*iad.Z21, np.sqrt(iad.gammaphiFG[i])*iad.Z02] for i in range(len(q_list))]
    
    #Kron prod just enough
    #Multi-qutrit Lindblad operators
    L = []
    for i in range(len(q_list)):
        L += [embedNoDouble(l1,q_list_num[i],len(q_list)) for l1 in L_1q[i]]
        
    #2-qutrit Linblad evolution matrix
    Ls = [sps.kron(Lm.conj(),Lm) - (1./2.)*sps.kron(sps.eye(np.shape(Lm)[0]),(Lm.conj().T).dot(Lm)) \
        - (1./2.)*sps.kron((Lm.conj().T).dot(Lm),sps.eye(np.shape(Lm)[0])) for Lm in L]
    Gm = 0.*sps.eye(np.shape(Ls[0])[0])
    for Lsi in Ls:
        Gm += Lsi
    return H_full, Gm

def getHandGExp(q_list,T,num_trot,decoh=True,dephas = True):
    """
    Enter q_list in terms of labels (Vinay/Machiel index)
    T: time to evolve H & G under
    num_trot: number of trotter steps to break T into
    """
    deltaT = T/num_trot
    
    #Set up qutrit indexing
    q_list_num = [labelToNum_dict[q] for q in q_list]
    num_q = len(q_list)
    pair_list = []
    #Get list of neighboring pairs that are contained in q_list
    for q1 in q_list_num:
        for q2 in q_list_num:
            if q1 == q2 + 1:
                pair_list.append(q2)
                
    #Figure out Hamiltonians And Hamiltonian exponential
    H_2 = [np.diag([0,0,0,   0,iad.alpha_11[i],iad.alpha_12[i],   0,iad.alpha_21[i],iad.alpha_22[i]]) for i in pair_list] #12 gain phase
    H_2_exp = [np.diag(np.exp(-1j*deltaT*np.array([0,0,0,   0,iad.alpha_11[i],iad.alpha_12[i],   0,iad.alpha_21[i],iad.alpha_22[i]]))) for i in pair_list] #12 gain phase
    #H_full = 0.*sps.eye(3**(2*num_q))
    H_full_exp = sps.eye(3**(2*num_q))
    for j in range(len(pair_list)):
        #H_full = H_full + embed2Ham(H_2[j],pair_list[j],num_q)
        H_full_exp = H_full_exp.dot(embed2Gate(H_2_exp[j],pair_list[j],num_q))    

    #1-qutrit Lindblad operators
    L_1q = [[ np.sqrt(iad.gamma1GE[i])*iad.lower10, np.sqrt(iad.gamma1EF[i])*iad.lower21, float(dephas)*np.sqrt(iad.gammaphiGE[i])*iad.Z10, float(dephas)*np.sqrt(iad.gammaphiEF[i])*iad.Z21, float(dephas)*np.sqrt(iad.gammaphiFG[i])*iad.Z02] for i in q_list_num]

    #Embed, multiply, embed, and add up 1-qutrit operators to get Lindbladian from decay only
    G_exp = sps.eye(3**(2*num_q))
    for i in range(len(q_list_num)):
        totalLi = 0*np.eye(3**(1 + num_q-1 + 1))
        for Lm in L_1q[i]:
            totalLi += np.kron(np.kron(Lm.conj(),np.eye(3**(num_q-1))),Lm) - \
                (1./2.)*np.kron(np.kron(np.eye(3),np.eye(3**(num_q-1))),np.dot(Lm.conj().T,Lm)) \
                - (1./2.)*np.kron(np.kron(np.dot(Lm.conj().T,Lm),np.eye(3**(num_q-1))),np.eye(3))
        G_exp = G_exp.dot(embedLKron(spla.expm(deltaT*totalLi),i,len(q_list))) #exponentiate term and multiply onto G
    
    #Multi-qutrit Lindblad operators - NOT NEEDED FOR EXPONENTIAL
#     L = []
#     for i in range(num_q):
#         L += [embedNoDouble(l1,q_list_num[i],len(q_list)) for l1 in L_1q[i]]
#     #2-qutrit Linblad evolution matrix
#     Ls = [sps.kron(Lm.conj(),Lm) - (1./2.)*sps.kron(sps.eye(np.shape(Lm)[0]),(Lm.conj().T).dot(Lm)) \
#         - (1./2.)*sps.kron((Lm.conj().T).dot(Lm),sps.eye(np.shape(Lm)[0])) for Lm in L]
#     Gm = 0.*sps.eye(np.shape(Ls[0])[0])
#     for Lsi in Ls:
#         Gm += Lsi
    
    U = sps.eye(3**(2*num_q))
    for i in range(num_trot):
        U = U.dot(H_full_exp)
        if decoh:
            U = U.dot(G_exp)
    
    return U

###
# Code to analyze error of H, G trotter decomposition as function of number of truncations - error not normalized very well, but 
# seems to gain a bit less after ~>5-10 trotter steps for times ~.5
###
# q_list = [1,0] #Only works for qutrits in a row right now - no skipping
# T = .5
# exactU = spla.expm(T*(H+G))

# trot = range(1,30)
# errors = []
# for num_trot in trot:
#     deltat = T/num_trot
#     H,G,He,Ge = getHandGExp(q_list,deltat,None)
#     U = sps.eye(3**(2*num_q))
#     for i in range(num_trot):
#         U = U.dot(He)
#         U = U.dot(Ge)
#     errors.append( np.sqrt((np.abs( U - exactU )**2).sum().sum()) / np.sqrt((np.abs( U )**2).sum().sum()) )
# import matplotlib.pyplot as plt
# plt.scatter(trot,np.log(errors))
# plt.show()

def actFullGateOnState(gate,rho_init,q_list,num_trot = 10,decoh=True,dephas = True):
    """
    gate: list with each entry either tuple of form (string, int) or float
    rho_init: initial density matrix in vector form OR identity MATRIX, if one wants the full unitary out
    q_list: qutrits to include in simulation - coupling to all other qutrits will be turned off
    num_trot: number of trotter steps to take in approximating exp(H + G)
    decoh: set to False to NOT include decoherence in simulation
    dephas: set to False to NOT include dephasing in simulation
    RUNTIME NOTES: dominated by time to find matrix exponentials, should be fixable; trotter size has much smaller effect
    """
    num_q = round(math.log(np.shape(rho_init)[0],3)/2) #determine number of qutrits from size of rho
    rho = rho_init
    time_evol_saved = {}
   

    for action in gate:
        if type(action) != tuple:
            if action in time_evol_saved.keys():
                evol = time_evol_saved[action]
                rho = evol.dot(rho)
            else:
                evol = getHandGExp(q_list,action,num_trot,decoh=decoh,dephas = dephas)
                rho = evol.dot(rho)
                time_evol_saved[action] = evol
        elif type(action) == tuple and len(action) == 2:
            rho = (embedGate(gate_dict[action[0]],action[1],num_q)).dot(rho) #embed gate to act on action[1]^th qubit, act
        elif type(action) == tuple and len(action) == 3: #single qutrit phase rotations
            gate = np.eye(3,dtype=complex)
            gate[action[0],action[0]] = np.exp(1j*action[2]) #set diagonal element to phase
            rho = (embedGate(gate,action[1],num_q)).dot(rho) #embed gate to act on action[1]^th qubit, act
    return rho

