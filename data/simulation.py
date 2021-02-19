from KG import simulate, KG_score, opt_KG_score
from LP import simulate_LP
import threading

T, alpha = 5, 1/4

KG = open("KG.txt", "w+")
opt_KG = open("opt-KG.txt", "w+")
LP = open("LP.txt", "w+")

KG.close()
opt_KG.close()
LP.close()

threads = []
for N in range(40, 1000, 40):
	N, M = N, 10 * N
	x = threading.Thread(target=simulate, args=(T, alpha, N, M, {(a, b): KG_score(a + 1, b + 1) for a in range(T) for b in range(T)}, "KG.txt"))
	x.start()
	
	y = threading.Thread(target=simulate, args=(T, alpha, N, M, {(a, b): opt_KG_score(a + 1, b + 1) for a in range(T) for b in range(T)}, "opt-KG.txt"))
	y.start()

	z = threading.Thread(target=simulate_LP, args=(T, alpha, N, M, "LP.txt"))
	z.start()
	
	x.join()
	y.join()
	z.join()
