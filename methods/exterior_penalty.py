import numpy as np

class ExteriorPenaltyMethod:
    def __init__(self, problem, solver, LIMIT):
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
        self.LIMIT = LIMIT
        
        self.timing = 0

    def get_timing(self):
        return self.timing

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
        cts = self.problem.cts
        mtx = flow
        ms_read, alphas_read, mtx_ct = cts
        mtx += mtx_ct

        print("flow mtx has %d nonzeros out of %d" % (np.count_nonzero(self.problem.flow), self.problem.flow.shape[0]*self.problem.flow.shape[1]))
        print("formula mtx has %d nonzeros out of %d" % (np.count_nonzero(mtx), mtx.shape[0]*mtx.shape[1]))
        
        initial = self.problem.initial()
        for i in range(self.LIMIT):
            print("External penalty iteration %d" % i)
            
            solution = self.solver.solve(mtx, initial)

            satisfied = self.problem.check(solution[0])
            if all(satisfied):
                # at the end of rounds, get timing for the entire run
                self.timing = self.solver.get_timing()
                print("External penalty has solution. Returning...")
                return solution
            else:
                for i in range(len(satisfied)):
                    print("ct%d: %s" % (i,satisfied[i]))
            
            mtx = flow
            # generate constraint matrix with updated weights
            ms_updated, mtx_ct_updated = self.problem.update_weights(solution[0])
            np.set_printoptions(threshold=np.inf)
            print("using ms: ", ms_updated)
            np.set_printoptions(threshold=6)
            mtx += mtx_ct_updated
            
            initial = solution
        
        print("External penalty has failed. Result is:")
        for i in range(len(satisfied)):
            print("ct%d: %s" % (i,satisfied[i]))
        return solution