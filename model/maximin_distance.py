import pulp
from pulp import PULP_CBC_CMD

from model.abstract_model import AbstractModel


class MaximinDistance(AbstractModel):
    def __init__(self, n: int):
        self.n = n
        self.model = pulp.LpProblem("MaximinDistance", pulp.LpMaximize)
        self.epsilon = 1e-6
        return

    def _set_iterables(self):
        self.cap_n = list(range(self.n))
        self.cap_n_prime = list(range(self.n - 1))
        self.cap_a = [(i, j) for i in self.cap_n for j in self.cap_n if i < j]
        return

    def _set_variables(self):
        self.x = pulp.LpVariable.dicts('x', self.cap_n, lowBound=0, upBound=1)
        self.y = pulp.LpVariable.dicts('y', self.cap_n, lowBound=0, upBound=1)
        self.d = pulp.LpVariable.dicts('d', self.cap_a, lowBound=0, upBound=2)
        self.u = pulp.LpVariable.dicts('u', self.cap_a, cat=pulp.LpBinary)
        self.w = pulp.LpVariable('w', lowBound=0, upBound=2)
        return

    def _set_objective(self):
        self.model += self.w
        return

    def _set_constraints(self):
        for i in self.cap_n_prime:
            self.model += (self.x[i] <= self.x[i + 1], f'monotone_{i}')
        for (i, j) in self.cap_a:
            self.model += (self.d[i, j] >= self.y[i] - self.y[j] + 2 *
                           (self.u[i, j] - 1), f'con1_{i}_{j}')
            self.model += (self.d[i, j] <= self.y[i] - self.y[j] + 2 *
                           (1 - self.u[i, j]), f'con2_{i}_{j}')
            self.model += (self.d[i, j] >=
                           self.y[j] - self.y[i] - 2 * self.u[i, j],
                           f'con3_{i}_{j}')
            self.model += (self.d[i, j] <=
                           self.y[j] - self.y[i] + 2 * self.u[i, j],
                           f'con4_{i}_{j}')
            self.model += (self.y[i] >= self.y[j] - 2 * (1 - self.u[i, j]),
                           f'con5_{i}_{j}')
            self.model += (self.y[i] <= self.y[j] + 2 * self.u[i, j],
                           f'con6_{i}_{j}')
            self.model += (self.w <= self.x[j] - self.x[i] + self.d[i, j],
                           f'dist_{i}_{j}')
        return

    def _optimize(self):
        time_limit_in_seconds = 1.5 * 60 * 60
        self.model.writeLP('test.lp')
        self.model.solve(PULP_CBC_CMD(timeLimit=time_limit_in_seconds))
        return

    def _is_feasible(self):
        return True

    def _process_infeasible_case(self):
        return list(), None

    def _post_process(self):
        coords = list()
        for i in self.cap_n:
            coords.append((self.x[i].value(), self.y[i].value()))
        return coords, self.w.value()
