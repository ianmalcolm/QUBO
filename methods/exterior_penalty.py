import numpy as np
from datetime import datetime

class ExteriorPenaltyMethod:
    def __init__(self, problem, solver, LIMIT, test_mode=False):
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
        self.test_mode = test_mode
        self.timing = 0

    def get_timing(self):
        return self.timing

    @staticmethod
    def random_filename():
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return "weights/weight_"+current_time+".txt"

    def run(self):
        '''
        repeat:
            squashes the flow and the constraints into a single matrix.
            passes the matrix to solver for solving.
            if result passes, return the result
            else, update penalty weight
        '''
        print("Running external penalty...")
        initial_flow = self.problem.flow
        ms_read, alphas_read, initial_ct = self.problem.cts
        initial_mtx = initial_flow + initial_ct

        print("flow mtx has %d nonzeros out of %d" % (np.count_nonzero(self.problem.flow), self.problem.flow.shape[0]*self.problem.flow.shape[1]))
        print("formula mtx has %d nonzeros out of %d" % (np.count_nonzero(initial_mtx), initial_mtx.shape[0]* initial_mtx.shape[1]))
        
        mtx = initial_mtx
        initial = self.problem.initial()
        for i in range(self.LIMIT):
            print("External penalty iteration %d" % i)
            
            solution = self.solver.solve(mtx, initial, test_mode=self.test_mode)
            if self.test_mode:
                for sol in solution:
                    configuration_to_check = sol[0]
                    satisfied = self.problem.check(configuration_to_check)
                    if all(satisfied):
                        self.timing = self.solver.get_timing()
                        print("External penalty has solution. Saving canonical weights...")
                        np.savetxt(ExteriorPenaltyMethod.random_filename(), ms_read, fmt='%.3f')
                        return configuration_to_check
                    else:
                        pass
            else:
                to_check = solution[0]
                satisfied = self.problem.check(to_check)
                if all(satisfied):
                    # at the end of rounds, get timing for the entire run
                    self.timing = self.solver.get_timing()
                    print("External penalty has solution. Saving canonical weights...")
                    np.savetxt(ExteriorPenaltyMethod.random_filename(), ms_read, fmt='%.3f')
                    return solution
                else:
                    for i in range(len(satisfied)):
                        print("ct%d: %s" % (i,satisfied[i]))
            
            if self.test_mode:
                best_solution = solution[0][0]
            else:
                best_solution = solution[0]
            # generate constraint matrix with updated weights
            self.problem.update_weights(best_solution)
            ms_read, alphas, updated_ct = self.problem.cts

            np.set_printoptions(threshold=np.inf)
            print("using ms: ", ms_read)
            np.set_printoptions(threshold=6)

            updated_mtx = initial_flow + updated_ct
            mtx = updated_mtx
            
            initial = solution
        
        self.timing = self.solver.get_timing()
        print("External penalty has failed. Result is:")
        for i in range(len(satisfied)):
            print("ct%d: %s" % (i,satisfied[i]))
        return solution