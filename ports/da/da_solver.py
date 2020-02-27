from .da_script_gen import DAScriptGen
import numpy as np
import utils.mtx as mt

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


class DASolver:
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
    
    def prepare_guidance_config(self, config_dict):
        ''' converts input dict of (int,int) into (str, boolean) '''
        ret = {}
        for k,v in config_dict.items():
            if v==1:
                ret[str(k)]=True
            else:
                ret[str(k)]=False
        return ret

    def solve(self, matrix, initial=()):
        mtx = matrix.copy()
        mtx = mt.to_upper_triangular(mtx)
        
        if initial:
            guidance_config = self.prepare_guidance_config(initial[0])
            print("guidance config:")
            print(guidance_config)
            #input()
            
        script_generator = DAScriptGen(API_KEY, CMD, mtx, SOLVER_NAME, guidance_config)
        script = script_generator.run()

        self.dequeue_if_full()

        # start and block until done
        subprocess.call(["./"+MAIN_SCRIPT])

        solution_dict = {}
        with open(RESPONSE_SCRIPT, 'r') as f:
            response = json.load(f)
            solutions = response['qubo_solution']['solutions']
            best_solution = solutions[0]
            print("best solution is : ")
            print(best_solution)
            best_config = best_solution["configuration"]
            best_energy = best_solution["energy"]
            for k,v in best_config.items():
                if v:
                    solution_dict[int(k)] = 1
                else:
                    solution_dict[int(k)] = 0
            
            timing = response['qubo_solution']['timing']['detailed']['anneal_time']
            self.timing += float(timing) / 1000

        return (solution_dict, best_energy)