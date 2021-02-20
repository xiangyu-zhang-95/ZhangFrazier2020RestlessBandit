import click
import numpy as np
import time
from gurobipy import *
import collections

class fluid_model():
    def p(self, s, a, sprime):
        if a == 0:
            return s == sprime
        if a == 1:
            tmp_a, tmp_b = s[0], s[1]
            tmp_c, tmp_d = sprime[0], sprime[1]
            if tmp_a == tmp_c and tmp_b + 1 == tmp_d:
                return 1 - (tmp_a + 1)/(tmp_a + tmp_b + 2)
            if tmp_a + 1 == tmp_c and tmp_b == tmp_d:
                return (tmp_a + 1)/(tmp_a + tmp_b + 2)
            return 0
    
    def reward(self, s, a):
        if a == 0:
            return 0.0
        else:
            return (s[0] + 1)/(s[0] + s[1] + 2)
    def __init__(self, T, alphas):
        self.alphas = alphas
        self.T = T
        self.m, self.x, self.y, self.z = None, {}, {}, {}
        self.duals = None
        self.A, self.B, self.C = set(), set(), set()

    def __LP_sol(self, T, alphas):
        
        time = [t for t in range(T)]
        win = [i for i in range(T)]
        los = [j for j in range(T)]
        d_var, reward = multidict({(t, i, j): (i+1)/(i+j+2) \
                                    for t in time \
                                        for i in win \
                                            for j in los})
        m = Model("MAB")
        n = m.addVars(time, win, los, name = "n")
        x = m.addVars(time, win, los, name = "x")
        
        # set ojective
        m.setObjective(sum(x[t, i, j] * reward[(t, i, j)] \
                            for t in time for i in win \
                                for j in los), GRB.MAXIMIZE)
        
        # add resource constraint
        m.addConstrs(sum(x[t, i, j] for i in win for j in los) \
                        == alphas[t] for t in time)
        
        # add initial constraint
        m.addConstr(n[0, 0, 0] == 1)
        m.addConstr(sum(n[0, i, j] for i in win for j in los) == 1)
        
        # add x constraint
        m.addConstrs(x[t, i, j] <= n[t, i, j] for t in time for i in win for j in los)
        m.addConstrs(x[t, i, j] >= 0 for t in time for i in win for j in los)
        
        # add fluid balance
        m.addConstrs(n[t, i, j] - x[t-1, i-1, j]*reward[(t-1, i-1, j)] - x[t-1, i, j-1]*(1-reward[(t-1, i, j-1)]) - (n[t-1, i, j] - x[t-1, i, j]) == 0 
                    for t in range(1, T) for i in range(1, T) for j in range(1, T))
        m.addConstrs(n[t, 0, j] - x[t-1, 0, j-1]*(1-reward[(t-1, 0, j-1)]) - (n[t-1, 0, j] - x[t-1, 0, j])== 0 
                    for t in range(1, T) for j in range(1, T))
        m.addConstrs(n[t, i, 0] - x[t-1, i-1, 0]*reward[(t-1, i-1, 0)] - (n[t-1, i, 0] - x[t-1, i, 0]) == 0
                    for t in range(1, T) for i in range(1, T))
        m.addConstrs(n[t, 0, 0] - (n[t-1, 0, 0] - x[t-1, 0, 0]) == 0 for t in range(1, T))
        m.setParam('OutputFlag', False)
        m.optimize()
        print('Obj: %g' % m.objVal)
        self.objVal = m.objVal
        
        return m

    def __solve(self, method=1):
        setParam("Method", method)
        T = self.T
        m = self.__LP_sol(self.T, self.alphas)
        self.m = m
        self.duals = m.PI[0: T]
        
        self.sorted = {t: [(a, b) for a in range(T) for b in range(T) if a + b <= t] for t in range(T)}
        v = {}
        for t in range(T - 1, -1, -1):
            advantage = {(a, b): 0 for a in range(T) for b in range(T) if a + b <= t}
            for a in range(T):
                for b in range(T):
                    if a + b <= t:
                        if t == T - 1:
                            r_pull = self.reward((a, b), 1) - self.duals[t]
                            r_idle = 0
                        else:
                            r_pull = self.reward((a, b), 1) - self.duals[t] + \
                                     self.p((a, b), 1, (a + 1, b)) * v[t + 1, (a + 1, b)] + \
                                     self.p((a, b), 1, (a, b + 1)) * v[t + 1, (a, b + 1)]
                            r_idle = v[t + 1, (a, b)]
                        advantage[(a, b)] = r_pull - r_idle
                        if r_pull < r_idle:
                            v[(t, (a, b))] = r_idle
                            continue
                        v[(t, (a, b))] = r_pull
            self.sorted[t].sort(reverse=True, key=lambda x: advantage[x])
        
        self.z[(0, (0, 0))] = 1
        self.pull_reward = 0
        for t in range(T):
            resource = self.alphas[t]
            for a, b in self.sorted[t]:
                self.x[(t, (a, b))] = min(self.z[(t, (a, b))], resource)
                self.pull_reward += self.x[(t, (a, b))] * self.reward((a, b), 1)
                self.y[(t, (a, b))] = self.z[(t, (a, b))] - self.x[(t, (a, b))]
                resource = resource - self.x[(t, (a, b))]
            if t == T - 1:
                continue
            for a, b in self.sorted[t + 1]:
                self.z[(t + 1, (a, b))] = (self.y[(t, (a, b))] if a + b <= t else 0) + \
                                          a / (a + b + 1) * (self.x[(t, (a - 1, b))] if a >= 1 else 0) + \
                                          b / (a + b + 1) * (self.x[(t, (a, b - 1))] if b >= 1 else 0)
    
    def calculate_occupation_measure_and_classify_state(self, epsilon=10**(-6)):
        self.__solve()
        m, sol, T = self.m, dict(), self.T

        for key in self.z:
            if self.x[key] > epsilon and self.y[key] < epsilon:
                self.A.add(key)
                continue
            if self.x[key] > epsilon and self.y[key] > epsilon:
                self.B.add(key)
                continue
            if self.x[key] < epsilon and self.y[key] > epsilon:
                self.C.add(key)
                continue
        return
    
    def check_degeneracy(self):
        if sorted([key[0] for key in self.B]) == [i for i in range(self.T)] and \
                    abs(self.objVal - self.pull_reward) < 10**(-13):
            print(f"""
            Model objective: {self.pull_reward}, different from {self.objVal} (solution from LP) less than 10e-15.
            Model is non-degenerate.
            """)
        else:
            raise Exception("Model is degenerate.\n")

    def calculate_diffusion_index(self, epsilon=10**(-8)):
        self.calculate_occupation_measure_and_classify_state()
        x, y, z, T = self.x, self.y, self.z, self.T
        v = {(t, (a, b)): 0 for t in range(T + 1) for a in range(t + 1) for b in range(t + 1) if a + b <= t}
        diffusion_index = {(t, (a, b)): 0 for t in range(T) for a in range(t + 1) for b in range(t + 1) if a + b <= t}

        for t in range(T - 1, -1, -1):
            state_t = [(a, b) for a in range(t + 1) for b in range(t + 1) if a + b <= t]
            for s in state_t:
                s_w = (s[0] + 1, s[1]) # state after win
                s_l = (s[0], s[1] + 1) # state after loss
                diffusion_index[(t, s)] = self.reward(s, 1) + self.p(s, 1, s_w)*v[(t + 1, s_w)] \
                                          + self.p(s, 1, s_l)*v[(t + 1, s_l)] \
                                          - self.reward(s, 0) - self.p(s, 0, s)*v[(t + 1, s)]
            
            l_y_g_0 = [(diffusion_index[(t, s)], s) for s in state_t if self.x[(t, s)] > epsilon]

            _, sbar = min(l_y_g_0)
            for s in state_t:
                if (t, s) in self.A:
                    v[(t, s)]= diffusion_index[(t, s)] - diffusion_index[(t, sbar)] + v[(t + 1, s)]
                else:
                    v[(t, s)] = self.reward(s, 0) + v[(t + 1, s)]
        self.diffusion_index = diffusion_index