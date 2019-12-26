import numpy as np

class ExteriorPenaltyMethod:
    def __init__(self, problem, solver):
        '''
            precondition:
                problem.isExterior = True
            
            Remark:
            - a solution is of the form (sample, energy)
        '''
        print("Initializing external penalty method...")
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
        print("Running external penalty...")
        flow = self.problem.flow.copy()
        mtx = flow
        ms = []
        alphas = []
        cts = []
        for m_0, alpha, ct in self.problem.cts:
            ms.append(m_0)
            alphas.append(alpha)
            cts.append(ct)
        for m_0, ct in zip(ms,cts):
            mtx += m_0 * ct

        print("flow mtx has %d nonzeros out of %d" % (np.count_nonzero(self.problem.flow), self.problem.flow.shape[0]*self.problem.flow.shape[1]))
        print("formula mtx has %d nonzeros out of %d" % (np.count_nonzero(mtx), mtx.shape[0]*mtx.shape[1]))
        LIMIT = 1
        initial = self.problem.initial()
        for i in range(LIMIT):
            print("External penalty iteration %d" % i)
            
            solution = self.solver.solve(mtx, initial)

            satisfied = self.problem.check(solution[0])
            if all(satisfied):
                print("External penalty has solution. Returning...")
                return solution
            
            mtx = flow
            print(type(mtx))
            for i in range(len(ms)):
                ms[i] = ms[i] * alphas[i]
            for m, ct in zip(ms, cts):
                print(type(m))
                print(type(ct))
                mtx += m*ct
            
            initial = solution
            first = False
        
        print("External penalty has failed. Result is:")
        for i in range(len(satisfied)):
            print("ct%d: %s" % (i,satisfied[i]))
        return solution