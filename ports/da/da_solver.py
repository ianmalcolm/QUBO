from .da_script_gen import DAScriptGen
from ..solver import Solver
import delete
import numpy as np
import utils.mtx as mt
import math


import subprocess
import json
from jinja2 import Environment, PackageLoader

API_KEY = "NB7wphN6nRhjE94wOhV9j7hWFJKKNI67"
SOLVER_NAME = "fujitsuDA2PT"
CMD = "async/qubo/solve"
MAIN_SCRIPT = "main.sh"
RESPONSE_SCRIPT = "response.txt"
JOB_ID_FILENAME = 'jobid.txt'
RESULT_FILENAME = 'da_result.txt'


class DASolver(Solver):
    def __init__(self):
        print("DA Solver created!")
        self.timing = 0

    def get_timing(self):
        return self.timing

    def dequeue_if_full(self):
        with open('jobs.txt','wb') as f:
            subprocess.call(["./jobs.sh"],stdout=f)

        env = Environment(loader=PackageLoader('ports.da','templates'))
        template = env.get_template('del.jinja')

        with open('jobs.txt', 'r') as f:
            jobs = json.load(f)['job_status_list']
            if len(jobs) > 15:
                j = jobs[0]
                jobid = j['job_id']
                with open('del.sh','w') as f:
                    del_rendered = template.render(job_id=jobid)
                    f.write(del_rendered)
                subprocess.call(['chmod', '+x', 'del.sh'])
                subprocess.call(['./del.sh'])
    
    def delete_if_full(self):
        with open('jobs.txt','wb') as f:
            subprocess.call(["./jobs.sh"],stdout=f)

        with open('jobs.txt', 'r') as f:
            jobs = json.load(f)['job_status_list']
            if len(jobs) > 15:
                delete.delete_all()

    def prepare_guidance_config(self, config_dict):
        ''' converts input dict of (int,int) into (str, boolean) '''
        ret = {}
        for k,v in config_dict.items():
            if v==1:
                ret[str(k)]=True
            else:
                ret[str(k)]=False
        return ret

    @staticmethod
    def temper(matrix):
        '''temper an input matrix
            input:
                matrix  symmetric matrix m
        '''
        mtx = matrix.copy()
        size = mtx.shape[0]
        # check if upper triangular. If so, make symmetric
        if np.allclose(mtx,np.triu(mtx)):
            to_add = mtx.T / 2
            np.fill_diagonal(to_add, 0)
            mtx += to_add
            mtx -= to_add.T

        n = int(math.sqrt(size))
        step = n
        for i in range(n):
            for j in range(n):
                window = mtx[i*step:(i+1)*step , j*step:(j+1)*step]
                # print(window)
                # input()
                mtx[i*step:(i+1)*step , j*step:(j+1)*step] -= np.average(window)
        
        for i in range(n):
            for j in range(n):
                list_row_indices = [k+i for k in range(0,n*n, step)]
                list_column_indices = [k+j for k in range(0,n*n, step)]
                window = mtx[list_row_indices, list_column_indices]
                # print(window)
                mtx[list_row_indices, list_column_indices] -= np.average(window)
        
        return mtx

    def solve(self, matrix, initial=(), test_mode=False):
        mtx = matrix.copy()
        mtx = DASolver.temper(mtx)
        mtx = mt.to_upper_triangular(mtx)
        np.savetxt("mtx.txt", mtx, fmt='%d')

        if initial:
            guidance_config = self.prepare_guidance_config(initial[0])
            #print("guidance config:")
            #print(guidance_config)
            #input()
        else:
            guidance_config = None
        params = super().sa_params(mtx)

        script_generator = DAScriptGen(API_KEY, CMD, mtx, SOLVER_NAME, params, guidance_config)
        script = script_generator.run()

        self.delete_if_full()

        # start and block until done
        subprocess.call(["./"+MAIN_SCRIPT])

        solution_dict = {}
        with open(RESPONSE_SCRIPT, 'r') as f:
            response = json.load(f)
            solutions = response['qubo_solution']['solutions']
            best_solution = solutions[0]
            #print("best solution is : ")
            #print(best_solution)
            best_config = best_solution["configuration"]
            best_energy = best_solution["energy"]
            for k,v in best_config.items():
                if v:
                    solution_dict[int(k)] = 1
                else:
                    solution_dict[int(k)] = 0
            
            if test_mode:
                self.timing = 0
            timing = response['qubo_solution']['timing']['detailed']['anneal_time']
            self.timing += float(timing) / 1000

        if not test_mode:
            return (solution_dict, best_energy)
        else:
            return self.to_solution_list(solutions)
    
    def to_solution_list(self, solutions):
        '''returns a list of samples
        Arg:
            solutions, a list of dicts
                'configuration'
                    node : bool
                'energy'
                'frequency'
        returns:
            a list of Samples
            a sample is (configuration, energy, frequency)
            a configuration is a dict of node:bool
        '''
        ret = []
        for solution in solutions:
            sample_configuration = {int(var):int(value) for var, value in solution['configuration'].items()}
            sample_energy = int(solution['energy'])
            sample_frequency = int(solution['frequency'])
            sample = (sample_configuration, sample_energy, sample_frequency)
            ret.append(sample)
        return ret