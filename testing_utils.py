import numpy as np
import yaml
import scipy as sc

def ideal_T1(qubit,time):

    data = load_decay_data()
    T1 = float(data['Q{}'.format(qubit)]['T1_GE'])

    return np.exp(-time/T1)

def ideal_T1EF(qubit, times):

    data = load_decay_data()
    T1 = float(data['Q{}'.format(qubit)]['T1_GE'])
    T1EF = float(data['Q{}'.format(qubit)]['T1_EF'])

    Gamma1 = 1./T1
    Gamma1EF = 1./T1EF

    decay_mat = np.array([[0, Gamma1, 0],
                          [0, -Gamma1, Gamma1EF],
                          [0, 0, -Gamma1EF]])

    init_state = np.array([0,0,1])
    
    return np.array([np.dot(sc.linalg.expm(decay_mat*time), init_state) for time in times])

def ideal_sq_Lindbladian(qubit):

    data = load_decay_data()
     
    T1 = float(data['Q{}'.format(qubit)]['T1_GE'])
    T1EF = float(data['Q{}'.format(qubit)]['T1_EF'])

    Tphi_GE = float(data['Q{}'.format(qubit)]['Tphi_GE'])
    Tphi_EF = float(data['Q{}'.format(qubit)]['Tphi_EF'])
    Tphi_FG = float(data['Q{}'.format(qubit)]['Tphi_FG']) 

     

def load_decay_data():

    with open('qutrit_decay.yaml', 'rb') as f:
        data = yaml.load(f)

    return data

