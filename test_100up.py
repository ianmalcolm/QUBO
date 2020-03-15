import numpy as np
from problems.placement import PlacementQAP
from ports.classical_simanneal import ClassicalNeal
from ports.da.da_solver import DASolver
import qaplib.readqaplib as qaplib
from methods.exterior_penalty import ExteriorPenaltyMethod
from sim.test_qap import QAPEvaluator
from methods.QAP import OurHeuristic

import os
import pandas as pd

def generate_filepath(filename, solver,timing):
    return os.path.join('data','100up', solver+'_'+filename+'_'+str(timing)+'.csv')

def postprocess(solution_list):
    columns=['config','energy','frequency']
    return pd.DataFrame(solution_list,columns=columns)

for filename in os.listdir('qaplib/100up'):
    if filename.endswith('.dat'):
        with open('data/100up/results.csv','a+') as f:
            F, D = qaplib.readqaplib(os.path.join('qaplib/100up',filename))
            size = F.shape[0]
            
            if size==100:
                k=10
            elif size==150:
                k=10

            evaluator = QAPEvaluator(size,size,F,D)
            method_da = OurHeuristic(
                size,
                size,
                k,
                F,
                D,
                fine_weight0=0,
                fine_alpha0=0,
                use_dwave_da_sw='da'
            )
            
            for i in range(6):
                solution_da = method_da.run()
                timing_da = method_da.get_timing()
                solution_da_df = postprocess(solution_da)
                filepath_da = generate_filepath(filename,'da',timing_da)
                solution_da_df.to_csv(filepath_da)
            
                all_energy_da = [evaluator.run(PlacementQAP.solution_matrix(sol[0],size,size)) for sol in solution_da]
                avg_energy_da = np.average(all_energy_da)
                std_energy_da = np.std(all_energy_da)
                #all_energy_sw = [evaluator.run(PlacementQAP.solution_matrix(sol[0],size,size)) for sol in solution_sw]
                #avg_energy_sw = np.average(all_energy_sw)
                #std_energy_sw = np.std(all_energy_sw)
                
                best_energy_da = evaluator.run(PlacementQAP.solution_matrix(solution_da[0][0],size,size))
                result_string = (filename +
                " " + str(best_energy_da) + 
                " " + str(avg_energy_da) + 
                " " + str(std_energy_da) +
                " " + str(timing_da) + '\n')
                f.write(result_string)