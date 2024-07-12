
# %%
with open('input.txt' , 'r') as f:
	file = f.readlines()

init = file[0]
n, k, t = tuple(map(int, init.split()))
file = file[1:]

h = [int(str) for str in file]

#print(n, k, t, h)

# %%
ht = h[t - 1]
#print('ht =', ht)

# %%
h.sort()
#print(h)

# %%
i1 = h.index(ht)
i2 = len(h) - h[::-1].index(ht) - 1
#print('i1, i2 = {}, {}'.format(i1, i2))

# %%
i_left = i2 - k + 1

if i_left < 0:
    i_left = 0

#print('i_left =', i_left)

# %%
d = []

if i_left > i1:
    d = [0]

for i in range(i_left, i1+1):
    i_right = i + k - 1
    if i_right > n - 1:
        break
    d.append(h[i_right] - h[i])

#print(d)

# %%
print(min(d))


