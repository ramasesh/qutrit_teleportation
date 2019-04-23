
import numpy as np
import scipy as sp
import scipy.linalg as spla

#Single qutrit gates acting on single qutrit
X = np.asarray([[0,1,0], [0,0,1], [1,0,0]])
omega = np.exp(1j*2*np.pi/3)
Z = np.diag([1,omega,omega.conj()])
flip_12 = np.asarray([[1,0,0], [0,0,1], [0,1,0]]) #Added single-qutrit Z such that all states gain same phase
flip_01 = np.asarray([[0,1,0], [1,0,0], [0,0,1]]) #Added single-qutrit Z such that all states gain same phase
flip_02 = np.asarray([[0,0,1], [0,1,0], [1,0,0]]) #Added single-qutrit Z such that all states gain same phase
Hadamard = 1/np.sqrt(3)*np.asarray([[1,1,1],[1, omega, omega.conj()],[1, omega.conj(), omega]]) #Takes X -> Z basis
Hadamard_a1 = 1/np.sqrt(3)*np.asarray([[1,1,omega],[1, omega, 1],[omega, 1,1]]) #Takes X -> XZ basis
Hadamard_a2 = 1/np.sqrt(3)*np.asarray([[1,1,omega.conj()],[1, omega.conj(), 1],[omega.conj(), 1,1]]) #Takes X -> ZX basis

gate_dict = {"flip12":flip_12, "flip01":flip_01, "flip02":flip_02, "X":X,"Xd":X.T, "Hadamard":Hadamard, "Hadamardc":Hadamard.conj()}

    