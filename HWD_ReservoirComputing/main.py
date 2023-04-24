import numpy as np
from scipy.integrate import solve_ivp

# ----------- 1) --------------
def lorenz(t, x):
    sigma = 10
    r = 28
    b = 8/3
    x1, x2, x3 = x
    y = x2
    Dx1 = -sigma*x1 + sigma*x2
    Dx2 = -x1*x3 + r*x1 - x2
    Dx3 = x1*y - b*x3
    return [Dx1, Dx2, Dx3]

x0 = np.array([0.1, -0.1, 0.05])  # Question 1: What initial conditions are expected?
end_time = 100
t_span = [0, end_time]
T = len(t_span)  # TODO: This yields the wrong value

sol = solve_ivp(lorenz, t_span, x0)
#print(sol)
#print(sol.y[1])
y = sol.y

# ----------- 2) --------------
class Reservoir:
    def __init__(self, n, N, w_std, gamma, y):
        self.n = n
        self.N = N
        self.w_std = w_std
        self.gamma = gamma
        self.y = np.array(y)

        self.x = np.hstack((x0.reshape((3, 1)), self.y))
        self.W = (np.random.rand(N, N) - 0.5) * w_std
        self.w_in = (np.random.rand(N, 3) - 0.5) * w_std
        self.w_out = (np.random.rand(3, N) - 0.5) * w_std
        self.r = np.zeros(N)
        self.R = np.zeros((N, T))

    def g(self, vec):
        return vec

    def train(self, num_generations):
        # Fill out R with reservoir values over time
        for t in range(T):
            reservoir_field = np.dot(self.W, self.r)
            input_field = np.dot(self.w_in, self.x[:, t])
            self.r = self.g(reservoir_field + input_field)
            self.R[:, t] = self.r
        a = 0



        

# Outside reservoir class
reservoir_size = 50  # Question 2: Did we get any advice on the size of the reservoir?
w_std = 0.1
tao = 3

reservoir = Reservoir(3, reservoir_size, w_std, tao, y)
reservoir.train(4)

