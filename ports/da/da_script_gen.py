from jinja2 import Environment, PackageLoader
import numpy as np
import json
import utils.mtx as mt

class DAScriptGen:
    def __init__(self, api_key, cmd, mtx, solver_name, params, guidance_config=None):
        self.api_key = api_key
        self.cmd = cmd
        self.mtx = mtx
        self.solver_name = solver_name
        self.guidance_config = guidance_config
        self.params = params
    
    def run(self):
        curl_data_dict = {}
        params = {}
        params['number_iterations'] = self.params['number_iterations']
        if self.solver_name == 'fujitsuDA2':
            params['number_runs'] = self.params['number_runs']
            params['offset_increase_rate'] = self.params['offset_increase_rate']
        elif self.solver_name == 'fujitsuDA2PT':
            params['number_replicas'] = self.params['number_replicas']

        if not self.guidance_config is None:
            params['guidance_config'] = self.guidance_config.copy()
        curl_data_dict[self.solver_name] = params
        
        qubo_matrix_dict = {}
        qubo_matrix_list = []
        size = self.mtx.shape[0]
        for i in range(size):
            row = mt.convert_to_int(list(self.mtx[i].astype(int)))
            row_dict = {}
            row_dict['qubo_row'] = row
            qubo_matrix_list.append(row_dict)
        qubo_matrix_dict['qubo'] = qubo_matrix_list
        curl_data_dict['qubo_matrix'] = qubo_matrix_dict        
        curl_data_rendered = json.dumps(curl_data_dict)
        with open('curl_data.txt','w') as f:
            f.write(curl_data_rendered)

        env = Environment(loader=PackageLoader('ports.da','templates'))
        template_file = env.get_template('da.jinja')
        da_rendered = template_file.render(
            api_key = self.api_key,
            cmd = self.cmd
        )
        with open('da_script.sh', 'w') as f:
            f.write(da_rendered)
        
        return da_rendered
    
    def run_jobid(self, job_id):
        env = Environment(loader=PackageLoader('ports.da','templates'))
        template = env.get_template('res.jinja')
        
        res_rendered = template.render(
            job_id = job_id
        )
        with open('res.sh','w') as f:
            f.write(res_rendered)