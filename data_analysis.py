import qaplib.readqaplib as qaplib
import os
import pandas as pd
import numpy as np
from sim.test_qap import QAPEvaluator
import utils.index as idx
import ast


def solution_matrix(solution,n,m):
    solution = ast.literal_eval(solution)
    solution_mtx = np.zeros((n, m), dtype=np.int8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            index = idx.index_1_q_to_l_1(i,j,m) - 1
            solution_mtx[i-1][j-1] = solution[index]
    return solution_mtx


def check_mtx(solution_mtx):
    size = solution_mtx.shape[0]
    test_ct1 = True
    test = np.zeros(size)
    for i in range(size):
        test += solution_mtx[i,:]
    result = test !=1
    if np.any(result):
        test_ct1 = False
    
    test_ct2 = True
    test = np.zeros(size)
    for i in range(size):
        test += solution_mtx[:,i]
    result = test != 1
    if np.any(result):
        test_ct2 = False
    
    return [test_ct1, test_ct2]

def collect_filenames():
    ret = {}
    for filename in os.listdir('data'):
        if filename.endswith('.csv'):
            components = filename.split('_')
            if not components[1] in ret:
                ret[components[1]] = [filename]
            else:
                ret[components[1]].append(filename)
    return ret

def get_best_energy(filename, dataset):
    best_energy = np.inf
    df = pd.read_csv(os.path.join('data',filename), index_col=0)
    F,D = problems[dataset]
    size = F.shape[0]
    evaluator = QAPEvaluator(size,size,F,D)
    for config in df['problem']:
        matrix = solution_matrix(config,size,size)
        if check_mtx(matrix):
            print("feasible solution found")
            energy = evaluator.run(matrix)
            if energy < best_energy:
                best_energy = energy
    return best_energy

filenames_dict = collect_filenames()
problems = {}
da_best_feasible_solutions = {}
sw_best_feasible_solutions = {}

for filename in os.listdir('qaplib'):
    if filename.endswith('.dat'):
        problems[filename] = qaplib.readqaplib(os.path.join('qaplib',filename))

# a list of dicts to be passed to pandas. each entry is one QAPLIB problem
final_record = []
    
for dataset, filename_list in filenames_dict.items():
    dataset_record = {}
    dataset_record['name'] = dataset
    da_time_list = []
    sw_time_list = []
    da_energy_list = []
    sw_energy_list = []
    for filename in filename_list:
        with open(os.path.join('data',filename), 'r') as f:
            components = filename.split('_')
            hardware = components[0]
            time = components[2][0:-4]
            
            if hardware=='da':
                da_time_list.append(float(time))
                da_energy = get_best_energy(filename, dataset)
                da_energy_list.append(int(da_energy))
            else:
                sw_time_list.append(float(time))
                sw_energy = get_best_energy(filename, dataset)
                sw_energy_list.append(int(sw_energy))

    print(da_time_list)
    print(da_energy_list)
    da_avg_time = np.average(da_time_list)
    dataset_record['da_avg_time'] = da_avg_time
    da_avg_energy = np.average(da_energy_list)
    dataset_record['da_avg_energy'] = da_avg_energy
    
    sw_avg_time = np.average(sw_time_list)
    dataset_record['sw_avg_time'] = sw_avg_time
    sw_avg_energy = np.average(sw_energy_list)
    dataset_record['sw_avg_energy'] = sw_avg_energy

    final_record.append(dataset_record)

final_record_df = pd.DataFrame(final_record)
final_record_df.to_csv('speed.csv')