import abc
import itertools
import numpy as np

class Solver(abc.ABC):
    '''
        Solver contains a solve() method that solves a QUBO matrix.
        
        Moreover, __init__ should provide the number of ancillaries for notational purpose.
        By default, ancillaries are the last few variables.
    '''
    @abc.abstractmethod
    def solve(self, mtx):
        pass

    @abc.abstractmethod
    def get_timing(self):
        pass

    def sa_params(self, mtx):
        ''' mtx is upper triangular '''
        params = {}
        params['number_iterations'] = 2000000000
        params['number_runs'] = 100
        params['number_replicas'] = 128
        size = mtx.shape[0]
        
        '''
        #compute default temperature range
        one_step_energy_changes = []
        for i in range(size):
            if mtx[i][i] !=0:
                one_step_energy_changes.append(abs(mtx[i][i]))
        for i,j in itertools.product(range(size),range(size)):
            if i<j and mtx[i][j]!=0:
                one_step_energy_changes.append(abs(mtx[i][j]))
        min_energy_change = min(one_step_energy_changes)
        
        energy_changes = {}
        for i in range(size):
            energy_changes[i] = abs(mtx[i][i])
        for i,j in itertools.product(range(size),range(size)):
            if i<j:
                energy_changes[i] += abs(mtx[i][j])
                energy_changes[j] += abs(mtx[i][j])
        max_energy_change = max(energy_changes.values())

            #at high temperature, min energy change should have 50% chance of being accepeted
        high_temp = -(min_energy_change / np.log(0.5))
            #at low temperature, max energy change should have a low chance, say 0.01, of being accepted
        low_temp = -(max_energy_change / np.log(0.01))

        params['temperature_start'] = high_temp
        params['temperature_interval'] = 1000

        temperature_range = [high_temp,low_temp]
        print(temperature_range)
        num_changes = params['number_iterations'] // params['temperature_interval']
        temperature_schedule = np.geomspace(*temperature_range, num_changes)
        temperature_decay = 1 - (temperature_schedule[1]/temperature_schedule[0])
        
        params['temperature_schedule'] = temperature_schedule
        params['offset_increase_rate'] = 10000
        params['temperature_decay'] = temperature_decay
        params['temperature_mode'] = 'EXPONENTIAL'
        params['temperature_range'] = temperature_range
        '''
        return params