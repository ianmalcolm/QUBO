import numpy as np
from problems.placement import PlacementQAP
from ports.classical_simanneal import ClassicalNeal
from ports.da.da_solver import DASolver
import qaplib.readqaplib as qaplib
from methods.exterior_penalty import ExteriorPenaltyMethod
from sim.test_qap import QAPEvaluator

import os
import pandas as pd

def generate_filepath(filename, solver,timing):
    return os.path.join('data', solver+'_'+filename+'_'+str(timing)+'.csv')

def postprocess(solution_list):
    columns=['problem','energy','frequency']
    return pd.DataFrame(solution_list,columns=columns)

def get_canonical_weights():
    ret = {}
    with open('canonical_weights.txt','r') as f:
        for line in f.read().splitlines():
            info = line.split()
            ret[info[0]] = float(info[1])
    return ret

canonical_weights_dict = get_canonical_weights()

for filename in os.listdir('qaplib'):
    if filename.endswith('tai30a.dat'):
        with open('data/speed_data.csv','a+') as f:
            F, D = qaplib.readqaplib(os.path.join('qaplib',filename))
            size = F.shape[0]
            
            evaluator = QAPEvaluator(size,size,F,D)
            problem = PlacementQAP(
                size,
                size,
                F,
                D,
                weight0=canonical_weights_dict[filename],
                alpha0=0
            )
            # solver_da = DASolver()
            solver_sw = ClassicalNeal()
            # method_da = ExteriorPenaltyMethod(problem,solver_da,1)
            method_sw = ExteriorPenaltyMethod(problem,solver_sw,1)
            
            columns=['problem','best_energy_da','time_da','best_energy_sw','time_sw']
            test_result_df = pd.DataFrame(columns=columns)
            for i in range(3):
                # solution_da = method_da.run(test_mode=True)
                # timing_da = method_da.get_timing()
                # solution_da_df = postprocess(solution_da)
                # filepath_da = generate_filepath(filename,'da',timing_da)
                # solution_da_df.to_csv(filepath_da)

                solution_sw = method_sw.run(test_mode=True)
                timing_sw = method_sw.get_timing()
                solution_sw_df = postprocess(solution_sw)
                filepath_sw = generate_filepath(filename,'sw',timing_sw)
                solution_sw_df.to_csv(filepath_sw)
            
                #all_energy_da = [evaluator.run(PlacementQAP.solution_matrix(sol[0],size,size)) for sol in solution_da]
                #avg_energy_da = np.average(all_energy_da)
                #std_energy_da = np.std(all_energy_da)
                #all_energy_sw = [evaluator.run(PlacementQAP.solution_matrix(sol[0],size,size)) for sol in solution_sw]
                #avg_energy_sw = np.average(all_energy_sw)
                #std_energy_sw = np.std(all_energy_sw)
                
                # best_energy_da = evaluator.run(PlacementQAP.solution_matrix(solution_da[0][0],size,size))
                best_energy_sw = evaluator.run(PlacementQAP.solution_matrix(solution_sw[0][0],size,size))

                result_string = (filename +
                # " " + str(best_energy_da) + 
                #" " + str(avg_energy_da) + 
                # " " + str(timing_da) + 
                " " + str(best_energy_sw) + 
                #" " + str(avg_energy_sw) + 
                " " + str(timing_sw) + '\n')
                f.write(result_string)