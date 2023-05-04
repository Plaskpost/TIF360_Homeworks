import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------- 1) --------------
def lorenz(t, x):
    sigma = 10
    r = 28
    b = 8/3
    x1, x2, x3 = x
    Dx1 = -sigma*x1 + sigma*x2
    Dx2 = -x1*x3 + r*x1 - x2
    Dx3 = x1*x2 - b*x3
    return [Dx1, Dx2, Dx3]

x0 = np.array([1.0, 1.0, 1.0])
end_time = 100
t_span = [0, end_time]

sol = solve_ivp(lorenz, t_span, x0)
Y = sol.y
T = np.shape(Y)[1]

# ----------- 2) --------------
class Reservoir:
    def __init__(self, n, N, sigma, x0, Y):
        self.n = n
        self.N = N
        self.sigma = sigma
        self.Y = np.array(Y)

        self.x = np.hstack((x0, self.Y))
        self.W = (np.random.rand(N, N) - 0.5) * sigma
        S = np.linalg.svd(self.W, compute_uv=False)
        self.W = self.W / max(S)
        self.w_in = (np.random.rand(N, n) - 0.5) * sigma
        self.w_out = (np.random.rand(n, N) - 0.5) * sigma

    def g(self, vec):
        return np.tanh(vec)

    def train(self, gamma):
        R = np.zeros((self.N, T))
        r = np.zeros(self.N)

        # Fill out R with reservoir values over time
        for t in range(T):
            reservoir_field = np.dot(self.W, r)
            input_field = np.dot(self.w_in, self.x[:, t])
            r = self.g(reservoir_field + input_field)
            R[:, t] = r

        # Set the w_out according to Bernhard's equation
        self.w_out = np.dot(np.dot(self.Y, R.transpose()), np.linalg.inv(np.dot(R, R.transpose()) + (gamma/2)*np.identity(self.N)))

    def predict(self, x0):
        x = x0
        r = np.zeros(self.N)
        y = np.zeros((self.n, T))
        for t in range(T):
            reservoir_field = np.dot(self.W, r)
            input_field = np.dot(self.w_in, x)
            r = self.g(reservoir_field + input_field.reshape(self.N))
            y[:, t] = np.dot(self.w_out, r)
            x = y[:, t]

        return y


# Outside reservoir class
reservoir_size = 1000
sigma = 0.2
gamma = 0.9

reservoir2 = Reservoir(n=3, N=reservoir_size, x0=x0.reshape((3, 1)), Y=Y, sigma=sigma)
reservoir2.train(gamma)
reservoir3 = Reservoir(1, reservoir_size, sigma, np.array(x0[1]).reshape((1,1)), (Y[1,:]).reshape((1,T)))
reservoir3.train(gamma)

r2_result_vec = reservoir2.predict(x0)
r3_result_vec = reservoir3.predict(x0[1])
lin_x = np.linspace(0, end_time, T)
plt.plot(lin_x, Y[1,:], label="Training data")
plt.plot(lin_x, r2_result_vec[1,:], label="Prediction from reservoir trained in 3 dimensions")
plt.plot(lin_x, r3_result_vec.reshape(T), label="Prediction from reservoir trained in 1 dimension")
plt.title("Predictions")
plt.legend()

# -------------- 3) ---------------
def time_to_divergence(reservoir_result, limit):
    t = 0
    while abs(reservoir_result[1, t] - Y[1, t]) < divergence_limit:
        t += 1
    return lin_x[t]

data_points = 50
divergence_limit = 3
singular_values = np.logspace(0, 100, data_points)
times_to_divergence = np.zeros((2, data_points))
plt.axvline(x=time_to_divergence(r2_result_vec, divergence_limit), color='red', linestyle='--')
plt.show()
W2 = reservoir2.W
W3 = reservoir3.W

for i in range(data_points):
    reservoir2.W = singular_values[i]*W2
    reservoir2.train(gamma)
    result2 = reservoir2.predict(x0)
    times_to_divergence[0, i] = time_to_divergence(result2, divergence_limit)
    reservoir3.W = singular_values[i]*W3
    reservoir3.train(gamma)
    result2 = reservoir2.predict(x0)
    times_to_divergence[1, i] = time_to_divergence(result2, divergence_limit)

plt.plot(singular_values, times_to_divergence[0,:], label="3 dimensional")
plt.plot(singular_values, times_to_divergence[1,:], label="1 dimensional")
plt.title("Time to divergence over maximal singular value.")
plt.legend()
plt.show()



