import numpy as np

X = np.zeros((100, 1000, 2))

with open('input.txt' , 'r') as f:
	for i in range(100):
		split = f.readline().split()
		X[i] = np.reshape(split, (1000, 2))


for i in range(100):
	if np.mean(np.abs(X[i])) < 0.37:
		print(1)
	else:
		print(2)