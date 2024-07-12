# %%
def v_dot_v(v1, v2):
    result_scalar = 0
    for i in range(len(v1)):
        result_scalar += v1[i]*v2[i]
    return result_scalar

def m_dot_v(matrix, vector):
    return [v_dot_v(v1, vector) for v1 in matrix]

def v_mean(list):
    m = 0
    for a in list:
        m += a
    return m/len(list)

# %%
class random:
    def __init__(self, seed=1):
        self.seed = seed
        return None
    def next_seed(self):
        self.seed = (self.seed * 73129 + 95121) % 100000
        return self.seed

    def random(self):
        return self.next_seed() / 100000
    def randint(self, limit=10):
        '''10000 max'''
        return self.next_seed() // 10 % limit

randomizer = random()

# %%
size = 1000

n_features = 15

x = [[randomizer.random() for x in range(n_features)] for y in range(size)]
w = [randomizer.random() for x in range(n_features)]

y = [1 + v_dot_v(v1, w) + .1*randomizer.random() for v1 in x]
#y = [1 + v_dot_v(v1, w) for v1 in x]

# %%
x_train, y_train = x, y

# %%
x_train = [[1] + x for x in x_train]
n_features += 1

# %% [markdown]
# ### GD без numpy

# %%
b = [randomizer.random() for i in range(n_features)]

print(b)

lr = .01
n_epochs = 5000
n_range = range(n_features)
size_range = range(size)

for epoch in range(n_epochs):

    y_pred = [v_dot_v(row, b) for row in x_train]

    error = [y_train[i] - y_pred[i] for i in size_range]

    sum_list = [0]*n_features
    for i, err in enumerate(error):
        for j in n_range:
            sum_list[j] += x_train[i][j] * err

    b_grad = [-2 * x / size for x in sum_list]

    b = [b[i] - lr * b_grad[i] for i in n_range]

    if epoch % (n_epochs/10) == 0:
        print('mse', v_mean([x**2 for x in error]))

print(1, w)
print(b)

# %% [markdown]
# ### SGD без numpy

# %%
b = [randomizer.random() for i in range(n_features)]

print(b)

lr = .01
n_epochs = 5000
feat_range = range(n_features)
size_range = range(size)

for epoch in range(n_epochs):
    r = randomizer.randint(size)
    x_i = x_train[r]
    y_i = y_train[r]
    y_pred = v_dot_v(x_i, b)

    error = y_i - y_pred

    b_grad = [-2 * i * error for i in x_i]

    b = [b[i] - lr * b_grad[i] for i in feat_range]

    if epoch % (n_epochs/10) == 0:
        error = [0]*size
        for i in size_range:
            error[i] = y_train[i] - v_dot_v(x_train[i], b)
        mse = v_mean([i**2 for i in error])
        print('mse', mse)

y_pred = [v_dot_v(x, b) for x in x_train]

print(1, w)
print(b)

# %% [markdown]
# ### Аналитическое решение с numpy

# %%
import numpy as np

# %%
class linear_regression:
    def __init__(self):
        return None
    def fit(self, x, y):
        X = np.array(x).astype('float64')
        Y = np.array(y).astype('float64')
        self.weight = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return True
    def predict(self, x):
        X = np.array(x)
        Y = X.dot(self.weight)
        return Y
    def mean_squared_error(self, y_true, y_pred):
        mse = np.average((y_true - y_pred) ** 2, axis=0)
        return mse

# %%
slice = size * 3 // 4
features_train = x_train[:slice]
target_train = y_train[:slice]
features_test = x_train[slice:]
target_test = y_train[slice:]

# %%
model = linear_regression()
model.fit(features_train, target_train)

predict = model.predict(features_test)

print('mse\n',model.mean_squared_error(target_test, predict))
print(1, w)
print(model.weight)


