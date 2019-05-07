import numpy as np
import scipy as sp
import scipy.sparse as sps

from functools import reduce

import sim_tools
import evolution
import state_prep

#Function to perform teleportation protocol
def getPandF(gatelist,
             psi,
             rho_EPR_07,
             rho_EPR_56,
             F_EPR_meas,
             decoh = True,
             dephas = True):
    """
    gatelist: list of gates
    psi: initial state to be teleported
    rho_EPR_07: 9 x 9 density matrix for 07 initial EPR pair
    rho_EPR_56: 9 x 9 density matrix for 56 initial EPR pair
    F_EPR_meas: fidelity of EPR measurement - currently assumes measurement is composition of depolarizing channel + perfect measurement, for simplicity
    decoh: Boolean
    RETURNS:
    P_psi: probability of successful EPR projection
    F_psi: fidelity of teleportation after EPR post-selection
    """
    ###
    # Setup initial state
    ###
    q_list = [1,0,7,6,5]
    rho_psi = sim_tools.vec_to_dm(psi)
    rho_init = reduce(sps.kron, [rho_psi, rho_EPR_07, rho_EPR_56]).reshape(-1, 1)
    ###
    # Evolve the state
    ###
    rho_final = evolution.actFullGateOnState(gatelist,
                                   rho_init,
                                   q_list ,
                                   num_trot = 10,
                                   decoh = decoh,
                                   dephas = dephas)
    ###
    # Get P and F from final state
    ###

    #EPR measurement = depolarizing channel + EPR projection
    #depolarizing - naively would guess this affects F (in ideal case) to be F_EPR_meas + (1-F_EPR_meas)*(1/9 + 8/9 * 1/3) - seems to do ~ a percent worse than this
    p = 1 - np.sqrt(F_EPR_meas) #sqrt because we are doing it twice - once on qutrit 0 and once on qutrit 7
    rho_final = evolution.depol_channel(rho_final,p,1) #0 & 7 labels are qutrits 1 & 2
    rho_final = evolution.depol_channel(rho_final,p,2)
    #Reshape rho
    rho_final = rho_final.reshape((243,243))
    #Perform EPR projection
    EPR_meas_full = reduce(sps.kron, [sps.eye(3), state_prep.rho_EPR_ideal, sps.eye(3**2)]) #proj_operator

    P_psi = rho_final.dot(EPR_meas_full).diagonal().sum()
    rho_postselect = EPR_meas_full.dot(rho_final.dot(EPR_meas_full))/P_psi

    rho_psi_5 = sps.kron(sps.eye(3**4),rho_psi) #projector operator
    F_psi = rho_postselect.dot(rho_psi_5).diagonal().sum()

    return P_psi,F_psi

def get_final_state(gatelist,
             psi,
             rho_EPR_07,
             rho_EPR_56,
             F_EPR_meas,
             qubits_to_measure = None,
             decoh = True,
             dephas = True,
             q_order = [1,0,7,6,5]):
    """
    gatelist: list of gates
    psi: initial state to be teleported
    rho_EPR_07: 9 x 9 density matrix for 07 initial EPR pair
    rho_EPR_56: 9 x 9 density matrix for 56 initial EPR pair
    F_EPR_meas: fidelity of EPR measurement - currently assumes measurement is composition of depolarizing channel + perfect measurement, for simplicity
    decoh: Boolean
    dephas: Boolean
    q_order: e.g. [1,0,7,6,5]
    qubits_to_measure: which qubits you want to look at
    RETURNS:
    P_psi: probability of successful EPR projection
    F_psi: fidelity of teleportation after EPR post-selection
    """
    rho_psi = sim_tools.vec_to_dm(psi)
    rho_init = reduce(sps.kron, [rho_psi, rho_EPR_07, rho_EPR_56]).reshape(-1, 1)
    ###
    # Evolve the state
    ###
    rho_final = evolution.actFullGateOnState(gatelist,
                                   rho_init,
                                   q_order,
                                   num_trot = 10,
                                   decoh = decoh,
                                   dephas = dephas)

    return rho_final.reshape(3**5, 3**5)

def print_report(Probabilities, Fidelities):
    P = [np.round(p, 3) for p in Probabilities]
    F = [np.round(f, 3) for f in Fidelities]
    print("P_psi = " + str(P))
    print("F_psi = " + str(F))
    print("<P> = " + str(np.array(Probabilities).mean()))
    print("<F> = {}".format(np.array(Fidelities).mean()))
    print("<FP>/<P> = " + str((np.array(Probabilities)*np.array(Fidelities)).mean()/np.array(Probabilities).mean()) + "\n\n")



