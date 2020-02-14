from dwave.system.samplers import DWaveSampler

sampler = DWaveSampler()
print(sampler.properties['annealing_time_range'])
print(sampler.properties['default_annealing_time'])
