import abc

class Problem(abc.ABC):
    '''
        An abstract problem contains a dict with:
        'flow'          the flow matrix
        'isExterior'    a boolean parameter specifying whether the problem is amenable to exterior penalty method.
                        if True, it has a non-zero 'alpha' and non-zero 'm_0's.
        'cts'           a list of constraint tuples. A constraint tuple is of the form (m_0, alpha, mtx)
            
        every matrix should have the same dimension.

    '''
    @abc.abstractmethod
    def check(self, solution):
        pass

    @abc.abstractproperty
    def flow(self):
        pass

    @abc.abstractproperty
    def isExterior(self):
        pass
    
    @abc.abstractproperty
    def cts(self):
        pass