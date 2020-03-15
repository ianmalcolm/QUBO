import json
import subprocess
from jinja2 import Environment, PackageLoader

def delete_all():
    with open('jobs.txt','wb') as f:
        subprocess.call(["./jobs.sh"],stdout=f)

    env = Environment(loader=PackageLoader('ports.da','templates'))
    template = env.get_template('del.jinja')


    with open('jobs.txt', 'r') as f:
        jobs = json.load(f)['job_status_list']
        for j in jobs:
            jobid = j['job_id']
            with open('del.sh','w') as f:
                del_rendered = template.render(job_id=jobid)
                f.write(del_rendered)
            subprocess.call(['chmod', '+x', 'del.sh'])
            subprocess.call(['./del.sh'])
        