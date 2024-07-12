import numpy as np

X = np.zeros((4, 3, 2))

with open('./input.txt' , 'r') as f:
	for i in range(4):
		l = list(map(int, f.readline().split()))
		X[i] = [[l[x], l[x+1]] for x in range(0, len(l), 2)]


for i in range(100):
	if np.mean(np.abs(X[i])) < 0.37:
		print(1)
	else:
		print(2)