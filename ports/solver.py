import abc

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