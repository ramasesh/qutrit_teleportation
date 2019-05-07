import numpy as np
import yaml
import scipy as sc

import sim_tools
import evolution
from functools import reduce

def ideal_T1(qubit,time):

    data = load_decay_data(qubit)
    T1 = float(data['T1_GE'])

    return np.exp(-time/T1)

def ideal_T1EF(qubit, times):

    data = load_decay_data(qubit)
    T1 = float(data['T1_GE'])
    T1EF = float(data['T1_EF'])

    Gamma1 = 1./T1
    Gamma1EF = 1./T1EF

    decay_mat = np.array([[0, Gamma1, 0],
                          [0, -Gamma1, Gamma1EF],
                          [0, 0, -Gamma1EF]])

    init_state = np.array([0,0,1])
    
    return np.array([np.dot(sc.linalg.expm(decay_mat*time), init_state) for time in times])

def ideal_Ramsey(qubit, times, detuning=1):
    """
    Currently only supports GE subspace  TODO   
    Currently does not take into account the T1 decay of the qubit, i.e. it only applies the decay to the coherences TODO
    """

    data = load_decay_data(qubit)
    T1 = float(data['T1_GE'])
    Tphi_GE = float(data['Tphi_GE'])


    Gamma2 = (1./(2*T1) + 1./(Tphi_GE))

    Epop = (1 + np.exp(-times*Gamma2) * np.cos(2*np.pi*detuning*times))/2.
    return Epop

def simulated_sq_idle(qubit, rho_init, delay_time):
    """ uses the single-qubit Lindbladian 
    (defined by the operators in interaction_and_decay.py) 
    to subject rho_init to idling for the specified delay_time
    
    Notes:
        rho_init should be in vectorized form"""

    data = load_decay_data(qubit)

    rho_out = evolution.actFullGateOnState([delay_time],
                                            rho_init,
                                            [qubit],
                                            num_trot=100)

    # rho_out will be in vectorized form, so:
    rho_out = sim_tools.vectorized_dm_to_dm(rho_out)

    return rho_out

def simulated_T1(qubit, delay_times, EF=False):
    """ uses the simulated_sq_idle() method to simulate a T1 experiment,
    for comparison to ideal theory calculations 
    
    Returns qubit populations as a 3 X N_pts array
    
    if EF is True, the initial state will be the F state
    if EF is False, the initial state will be the E state"""
    
    n_pts = len(delay_times)
    n_lvls = 3

    populations = np.zeros((n_lvls, n_pts))

    if not EF:
        rho_init = sim_tools.vectorize_dm(np.diag([0, 1., 0]))
    else:
        rho_init = sim_tools.vectorize_dm(np.diag([0, 0, 1.]))

    for idx, time in enumerate(delay_times):
        rho_after = simulated_sq_idle(qubit,
                                      rho_init,
                                      time)
        populations[:, idx] = np.diag(rho_after)

    return populations

def simulated_Ramsey(qubit, delay_times, subspace='GE', detuning=1):
    """ uses the simulated_sq_idle() method to simulate a Ramsey experiment,
    for comparison to ideal theory calculations

    Returns qubit populations (after the final pi/2 pulse) as a 3 X N_pts array
    Argument 'detuning' is the artificial detuning in the Ramsey experiment, units of MHz"""

    n_pts = len(delay_times)
    n_lvls = 3

    populations = np.zeros((n_lvls, n_pts))

    if subspace == 'GE' or subspace == 'EG':
        psi_init = np.array([1., 1., 0], dtype='complex')
    elif subspace == 'EF' or subspace == 'FE':
        psi_init = np.array([0, 1., 1.], dtype='complex')
    elif subspace == 'GF' or subspace == 'FG':
        psi_init = np.array([1., 0, 1.], dtype='complex')
    psi_init = psi_init / np.linalg.norm(psi_init, ord=2)

    rho_init = sim_tools.vec_to_dm(psi_init)
    rho_init = sim_tools.vectorize_dm(rho_init)

    for idx, time in enumerate(delay_times):
        U = reduce(np.dot, [sim_tools.rot_z(2*np.pi*detuning*time),
                            sim_tools.rot_y(np.pi/2), 
                            sim_tools.rot_z(-2*np.pi*detuning*time)])

        U = sim_tools.bit_to_trit(U, subspace=subspace)
        
        rho_after = simulated_sq_idle(qubit,
                                      rho_init,
                                      time)

        rho_after = reduce(np.dot, [U, rho_after, np.conj(U.T)])
        
        populations[:, idx] = np.real(np.diag(rho_after))
    
    return populations

def load_decay_data(qubit):

    with open('qutrit_decay.yaml', 'rb') as f:
        data = yaml.load(f)

    return data['Q{}'.format(qubit)]

def test_depol(p):
    """ Sanity check on the depolarizing channel implemented in evolution.py,
    specifically checking that two applications of the channel with probability 'p' each 
    is equivalent to a single application with probability '2p - p^2' """

    """ sample a random single-qutrit pure state """ 
    psi = sim_tools.random_pure_state()
    rho = sim_tools.vec_to_dm(psi)
    rho_vec = sim_tools.vectorize_dm(rho)

    rho_depol1 = evolution.depol_channel(rho_vec, p, 0)
    rho_depol1 = evolution.depol_channel(rho_depol1, p, 0)

    rho_depol2 = evolution.depol_channel(rho_vec, 2*p - p*p, 0)

    print('Difference (should be zero)')
    print(np.linalg.norm(rho_depol1 - rho_depol2, ord=2))
    
