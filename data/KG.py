from scipy.stats import beta
import numpy as np
import collections
import time

def I(a, b):
    return beta.cdf(1/2, a, b)

def h(x):
    return max(x, 1 - x)

def R1(a, b):
    return h(I(a + 1, b)) - h(I(a, b))

def R2(a, b):
    return h(I(a, b + 1)) - h(I(a, b))

def opt_KG_score(a, b):
    """
    KG score function
    """
    return max(R1(a, b), R2(a, b))

def KG_score(a, b):
    return a / (a + b) * R1(a, b) + b / (a + b) * R2(a, b)

# pull arm from period 0 to period T - 1.
def Priority_policy(T, score, alpha=1/4, initial=collections.defaultdict(float, {(0, 0): 1.0})):
    curr_state = initial
    for _ in range(T):
        #print("curr state: ", curr_state)
        next_state = collections.defaultdict(float)
        budget = alpha
        sorted_state = sorted(curr_state.keys(), key=lambda x: score(x[0] + 1, x[1] + 1), reverse=True)
        #print("sorted state: ", sorted_state)
        for s in sorted_state:
            pulled = min(curr_state[s], budget)
            #print("pulled ", pulled, " of state ", s, " with alpha=", budget)
            budget = budget - pulled
            a, b = s
            next_state[(a + 1, b)] += pulled * (a + 1) / (a + b + 2)
            next_state[(a, b + 1)] += pulled * (b + 1) / (a + b + 2)
            next_state[(a, b)] += curr_state[s] - pulled
        curr_state = next_state
    
    def rewards(a, b):
        if a < b:
            return beta.cdf(1/2, a + 1, b + 1)
        return 1 - beta.cdf(1/2, a + 1, b + 1)
    
    return sum([curr_state[s] * rewards(s[0], s[1]) for s in curr_state])

def opt_KG_policy(T, alpha):
    return Priority_policy(T=T, score=opt_KG_score, alpha=alpha, initial=collections.defaultdict(float, {(0, 0): 1.0}))

def KG_policy(T, alpha):
    return Priority_policy(T=T, score=KG_score, alpha=alpha, initial=collections.defaultdict(float, {(0, 0): 1.0}))

def reward(a, b):
	if a < b:
		return beta.cdf(1/2, a + 1, b + 1)
	return 1 - beta.cdf(1/2, a + 1, b + 1)

def simulate(T, alpha, N, M, score, file):
    start = time.time()
    rewards = []
    for _ in range(M):
        curr_state = {(0, 0): N}
        for __ in range(T):
            next_state = collections.defaultdict(int)
            budget = int(alpha * N)
            sorted_state = sorted(curr_state.keys(), key=lambda x: score[x], reverse=True)
            for s in sorted_state:
                pulled = min(curr_state[s], budget)
                budget = budget - pulled
                a, b = s
                tmp = np.random.binomial(pulled, (a + 1) / (a + b + 2))
                next_state[(a + 1, b)] += tmp
                next_state[(a, b + 1)] += pulled - tmp
                next_state[(a, b)] += curr_state[s] - pulled
            curr_state = next_state
        rewards.append(sum([curr_state[s] * reward(s[0], s[1]) for s in curr_state]))
    end = time.time()
    f = open(file, 'a')
    mean, std = np.mean(rewards), np.std(rewards) / np.sqrt(M)
    f.write(f"N:{N} M:{M} mean:{mean} std:{std} time:{end - start}\n")
    f.close()

