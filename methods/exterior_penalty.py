class ExteriorPenaltyMethod:
    def __init__(self, problem, solver):
        '''
            precondition:
                problem.isExterior = True
        '''
        if not problem.isExterior:
            raise ValueError("cannot solve a non-exterior problem using exterior penalty method")
        self.problem = problem
        self.solver = solver
        
    def run(self):
        '''
        repeat:
            squashes the flow and the constraints into a single matrix.
            passes the matrix to solver for solving.
            if result passes, return the result
            else, update penalty weight
        '''
        flow = self.problem.flow
        mtx = flow
        ms = []
        alphas = []
        cts = []
        for m_0, alpha, ct in self.problem.cts:
            ms.append(m_0)
            alphas = alpha
            cts = ct
        for m_0, ct in zip(ms,cts):
            mtx += m_0 * ct

        LIMIT = 1
        for i in range(LIMIT):
            solution = self.solver.solve(mtx)
            satisfied = self.problem.check(solution)
            if satisfied:
                return solution
            
            mtx = flow
            for i in range(len(ms)):
                ms[i] = ms * alphas[i]
            for m, ct in zip(ms, cts):
                mtx += m*ct
        
        return solution