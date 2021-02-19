from scipy.stats import beta
from gurobipy import *
import numpy as np
import collections
import time

# pull arms from start to T - 1. test at T.
def LP_sol(T, start=0, initial={(0, 0): 1.0}, alpha = 1/4, print_full_information=False):
    def transit(t, a, b):
        return (a + 1)/(a + b + 2)
    
    def rewards(a, b):
        if a < b:
            return beta.cdf(1/2, a + 1, b + 1)
        return 1 - beta.cdf(1/2, a + 1, b + 1)
    T = T + 1
    time = [t for t in range(start, T)]
    win = [i for i in range(T)]
    los = [j for j in range(T)]
    d_var, reward = multidict({(t, i, j): (0 if t < T-1 else rewards(i, j)) for t in time for i in win for j in los})
    m = Model("crowdsourcing")
    n = m.addVars(time, win, los, name = "n")
    x = m.addVars(time, win, los, name = "x")
    
    # set ojective
    # m.setObjective(sum(n[t, i, j] * reward[(t, i, j)] for t in time for i in win for j in los), GRB.MAXIMIZE)
    m.setObjective(sum(n[T - 1, i, j] * reward[(T - 1, i, j)] for i in win for j in los), GRB.MAXIMIZE)
    
    
    # add resource constraint
    m.addConstrs(sum(x[t, i, j] for i in win for j in los) == alpha for t in time)
    
    # add initial constraint
    for t in initial:
        m.addConstr(n[start, t[0], t[1]] == initial[t])
    m.addConstr(sum(n[start, i, j] for i in win for j in los) == 1)
    
    # add x constraint
    m.addConstrs(x[t, i, j] <= n[t, i, j] for t in time for i in win for j in los)
    m.addConstrs(x[t, i, j] >= 0 for t in time for i in win for j in los)
    
    # add fluid balance
    m.addConstrs(n[t, i, j] - x[t-1, i-1, j]*transit(t-1, i-1, j) - x[t-1, i, j-1]*(1-transit(t-1, i, j-1)) - (n[t-1, i, j] - x[t-1, i, j]) == 0 
                 for t in range(start + 1, T) for i in range(1, T) for j in range(1, T))
    m.addConstrs(n[t, 0, j] - x[t-1, 0, j-1]*(1-transit(t-1, 0, j-1)) - (n[t-1, 0, j] - x[t-1, 0, j])== 0 
                 for t in range(start + 1, T) for j in range(1, T))
    m.addConstrs(n[t, i, 0] - x[t-1, i-1, 0]*transit(t-1, i-1, 0) - (n[t-1, i, 0] - x[t-1, i, 0]) == 0
                 for t in range(start + 1, T) for i in range(1, T))
    m.addConstrs(n[t, 0, 0] - (n[t-1, 0, 0] - x[t-1, 0, 0]) == 0 for t in range(start + 1, T))
    m.setParam( 'OutputFlag', print_full_information)
    m.optimize()
    return m

def occupation_measure(T, alpha = 1/4):
    m = LP_sol(T, alpha=alpha)
    sol = dict()
    for v in m.getVars():
        sol[v.varName] = v.x
    Z = {t: {} for t in range(T)}
    X = {t: {} for t in range(T)}
    Y = {t: {} for t in range(T)}
    for t in range(T):
        for a in range(T):
            for b in range(T):
                name_n = "n[" + str(t) + ',' + str(a) + ',' + str(b) + ']'
                name_x = "x[" + str(t) + ',' + str(a) + ',' + str(b) + ']'
                Z[t][(a, b)] = sol[name_n]
                X[t][(a, b)] = sol[name_x]
                Y[t][(a, b)] = sol[name_n] - sol[name_x]
    return X, Y, Z

def to_category(X, Y, Z):
    T = len(X)
    HP, MP, LP = {t:[] for t in range(T)}, {t:[] for t in range(T)}, {t:[] for t in range(T)}
    for t in range(T):
        for s in X[t]:
            if X[t][s] > 0 and Y[t][s] == 0:
                HP[t].append(s)
                continue
            if X[t][s] == 0 and Y[t][s] > 0:
                LP[t].append(s)
                continue
            if X[t][s] > 0 and Y[t][s] > 0:
                MP[t].append(s)
                continue
    return HP, MP, LP

def simulate_LP(T, alpha, N, M, file):
    start = time.time()
    def reward(a, b):
        if a < b:
            return beta.cdf(1/2, a + 1, b + 1)
        return 1 - beta.cdf(1/2, a + 1, b + 1)

    X, Y, Z = occupation_measure(T, alpha)
    HP, MP, LP = to_category(X, Y, Z)
    rewards = []
    for _ in range(M):
        current_state = {(0, 0): N}
        for t in range(T):
            next_state = collections.defaultdict(int)
            budget = int(alpha * N)
            undecide = collections.defaultdict(int)

            for s in HP[t]:
                pulled = min(current_state[s], budget)
                budget -= pulled
                a, b = s
                tmp = np.random.binomial(pulled, (a + 1) / (a + b + 2))
                next_state[(a + 1, b)] += tmp
                next_state[(a, b + 1)] += pulled - tmp
                next_state[(a, b)] += current_state[s] - pulled
            for s in MP[t]:
                pulled = min(current_state[s], budget, N * X[t][s])
                budget -= pulled
                a, b = s
                tmp = np.random.binomial(pulled, (a + 1) / (a + b + 2))
                next_state[(a + 1, b)] += tmp
                next_state[(a, b + 1)] += pulled - tmp
                undecide[s] = current_state[s] - pulled
            for s in MP[t]:
                pulled = min(undecide[s], budget)
                budget -= pulled
                a, b = s
                tmp = np.random.binomial(pulled, (a + 1) / (a + b + 2))
                next_state[(a + 1, b)] += tmp
                next_state[(a, b + 1)] += pulled - tmp
                next_state[(a, b)] += undecide[s] - pulled
            for s in LP[t]:
                pulled = min(current_state[s], budget)
                budget -= pulled
                a, b = s
                tmp = np.random.binomial(pulled, (a + 1) / (a + b + 2))
                next_state[(a + 1, b)] += tmp
                next_state[(a, b + 1)] += pulled - tmp
                next_state[(a, b)] += current_state[s] - pulled
            current_state = next_state
        rewards.append(sum(reward(s[0], s[1]) * current_state[s] for s in current_state))

    end = time.time()
    f = open(file, 'a')
    mean, std = np.mean(rewards), np.std(rewards) / np.sqrt(M)
    f.write(f"N:{N} M:{M} mean:{mean} std:{std} time:{end - start}\n")
    f.close()
