import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

def embedGate(gate_single,i,num_q):
    """embeds single qutrit gate on qutrit i into a many qutrit gate on density matrix of N qutrits (acting as identity on all other qutrits)"""
    gate = sps.csr_matrix(gate_single)
    for j in range(0,i):
        gate = sps.kron(sps.eye(3),gate)
    for j in range(i+1,num_q):
        gate = sps.kron(gate,sps.eye(3))
    gate = sps.kron(gate,gate.conj()) #for density matrix evolution
    return gate

def embed2Ham_notsparse(gate_2,i,num_q):
    """embeds single qutrit gate on qutrit i & i+1 into a many qutrit gate on density matrix of num_q qutrits (acting as identity on all other qutrits)"""
    gate = gate_2
    for j in range(0,i):
        gate = np.kron(np.eye(3),gate)
    for j in range(i+2,num_q):
        gate = np.kron(gate,np.eye(3))
    gate = -1j*(np.kron(gate,np.eye(np.shape(gate)[0])) - np.kron(np.eye(np.shape(gate)[0]),gate))
    return gate

def embedNoDouble_notsparse(gate_single,i,num_q):
    """embeds single qutrit gate on qutrit i into a many qutrit gate on density matrix of N qutrits (acting as identity on all other qutrits)"""
    gate = gate_single
    for j in range(0,i):
        gate = np.kron(np.eye(3),gate)
    for j in range(i+1,num_q):
        gate = np.kron(gate,np.eye(3))
    return gate

def embed2Ham(gate_2,i,num_q):
    """embeds two qutrit Hamiltonian on qutrit i & i+1 into a many qutrit gate on density matrix of num_q qutrits (acting as identity on all other qutrits)"""
    gate = sps.csr_matrix(gate_2)
    for j in range(0,i):
        gate = sps.kron(sps.eye(3),gate)
    for j in range(i+2,num_q):
        gate = sps.kron(gate,sps.eye(3))
    gate = -1j*(sps.kron(gate,sps.eye(np.shape(gate)[0])) - sps.kron(sps.eye(np.shape(gate)[0]),gate))
    return gate

#1 ms for 4 qutrits
def embed2Gate(gate_2,i,num_q):
    """embeds two qutrit Hamiltonian on qutrit i & i+1 into a many qutrit gate on density matrix of num_q qutrits (acting as identity on all other qutrits)"""
    gate = sps.csr_matrix(gate_2)
    for j in range(0,i):
        gate = sps.kron(sps.eye(3),gate)
    for j in range(i+2,num_q):
        gate = sps.kron(gate,sps.eye(3))
    gate = sps.kron(gate,gate.conj())
    return gate

def embedLKron(gate_single,i,num_q):
    """a bit funky: embeds 3**(1 + (num_q - 1) + 1)-d L matrix into 3**(2*num_q) matrix"""
    gate = sps.csr_matrix(gate_single)
    for j in range(0,i):
        gate = sps.kron(sps.eye(3),gate)
    for j in range(i+1,num_q):
        gate = sps.kron(gate,sps.eye(3))
    return gate

def embedNoDouble(gate_single,i,num_q):
    """embeds single qutrit gate on qutrit i into a many qutrit gate on density matrix of N qutrits (acting as identity on all other qutrits)"""
    gate = sps.csr_matrix(gate_single)
    for j in range(0,i):
        gate = sps.kron(sps.eye(3),gate)
    for j in range(i+1,num_q):
        gate = sps.kron(gate,sps.eye(3))
    return gate