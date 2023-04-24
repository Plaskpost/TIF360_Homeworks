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

x0 = [0.1, -0.1, 0.05]  # Question 1: What initial conditions are expected?
T = 100
t_span = [0, T]

sol = solve_ivp(lorenz, t_span, x0)
#print(sol)
#print(sol.y[1])
y = sol.y[1]

# ----------- 2) --------------
class Reservoir:
    def __init__(self, N, reservoir_size, w_std, tao, y):
        self.N = N
        self.reservoir_size = reservoir_size
        self.w_std = w_std
        self.tao = tao
        self.y = np.array(y)

        self.x = self.y[0]
        self.W = (np.random.rand(reservoir_size, reservoir_size) - 0.5) * w_std
        self.w_in = (np.random.rand(reservoir_size, 3) - 0.5) * w_std
        self.w_out = (np.random.rand(3, reservoir_size) - 0.5) * w_std
        self.r = np.zeros(reservoir_size)

    def g(self, vec):
        return vec

    def train(self, num_generations):
        for gen in range(0,num_generations):
            # Reset something
            for t in range(1,T):
                self.r = self.g(np.dot(self.W,self.r) + np.dot(self.w_in,self.x))
                O = np.dot(self.w_out,self.r)
                H = (1/2)*(np.dot((self.y[t]-O),(self.y[t]-O))) # TODO: Sum over time

                # Update to next input data
                self.x = self.y[t]

        

# Outside reservoir class
reservoir_size = 50  # Question 2: Did we get any advice on the size of the reservoir?
num_generations = 100
w_std = 0.1
tao = 3

reservoir = Reservoir(3, reservoir_size, w_std, tao, y)
reservoir.train(4)

