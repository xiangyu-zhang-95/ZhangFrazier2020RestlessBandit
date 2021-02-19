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

for a, b in [(40, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]:
	threads = []
	for N in range(a, b, 40):
		N, M = N, 10 * N
		x = threading.Thread(target=simulate, args=(T, alpha, N, M, KG_score, "KG.txt"))
		x.start()
		threads.append(x)
		
		y = threading.Thread(target=simulate, args=(T, alpha, N, M, opt_KG_score, "opt-KG.txt"))
		y.start()
		threads.append(y)

		z = threading.Thread(target=simulate_LP, args=(T, alpha, N, M, "LP.txt"))
		z.start()
		threads.append(z)

	for t in threads:
		t.join()

