import numpy as np
import os
from datetime import datetime

def random_filename():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "res_"+current_time+".dat"

class BigArray:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        
        filename = random_filename()
        filepath = os.path.join('xujun',filename)
        self.fp = np.memmap(filepath, dtype=self.dtype, mode='w+', shape=self.shape)

    def get_array(self):
        return self.fp

    @staticmethod
    def get_size_GB(shape, bitwidth):
        n,m = shape
        num_bytes = n*m*(bitwidth / 8)
        return num_bytes / (1024*1024*1024)