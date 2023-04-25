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

sol = solve_ivp(lorenz, t_span, x0)
#print(sol)
#print(sol.y[1])
Y = sol.y
T = np.shape(Y)[1]

# ----------- 2) --------------
class Reservoir:
    def __init__(self, n, N, sigma, gamma, Y):
        self.n = n
        self.N = N
        self.sigma = sigma
        self.gamma = gamma
        self.Y = np.array(Y)

        self.x = np.hstack((x0.reshape((3, 1)), self.Y))
        self.W = (np.random.rand(N, N) - 0.5) * sigma
        self.w_in = (np.random.rand(N, 3) - 0.5) * sigma
        self.w_out = (np.random.rand(3, N) - 0.5) * sigma
        self.r = np.zeros(N)

    def g(self, vec):
        return vec

    def train(self):
        R = np.zeros((self.N, T))

        # Fill out R with reservoir values over time
        for t in range(T):
            reservoir_field = np.dot(self.W, self.r)
            input_field = np.dot(self.w_in, self.x[:, t])
            self.r = self.g(reservoir_field + input_field)
            R[:, t] = self.r

        # Set the w_out according to Bernhard's equation
        self.w_out = self.Y * (R * R.transpose() + )  # : Finish this equation



        

# Outside reservoir class
reservoir_size = 50  # Question 2: Did we get any advice on the size of the reservoir?
sigma = 0.1
tao = 3

reservoir = Reservoir(3, reservoir_size, sigma, tao, y)
reservoir.train(4)

