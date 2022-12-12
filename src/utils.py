import numpy as np
import torch as tc
import matplotlib.pyplot as plt
device0 = "cuda:0"


def plot_vars(self, l):
    def func(x):
        self.S_paths.reset_tensors(l, self)
        self.S_paths.theta = self.S_paths.theta + x
        self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l])
        oVar, E = self.S_paths.var_mean(option=self.option)
        return oVar

    def func2(x):
        self.S_paths.reset_tensors(l, self)
        self.S_paths.theta = x.fill_(0.0)
        self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l])
        oVar, E = self.S_paths.var_mean(option=self.option)
        return oVar

    plotshow_func(func2, color="green")
    plotshow_func(func, color="blue", mark= self.Theta[l])
    plt.show()

def plotshow_func(func, start = -2, stop = 2, color = "red", labely = "", labelx = "", mark = None):
    import matplotlib.pyplot as plt
    import numpy as np
    n = 50
    step = (stop - start) / n
    x = np.linspace(start = start, stop = stop, num=n)
    y = np.zeros_like(x)
    for i in range(len(x)):
        x_tensor = tc.tensor([x[i]], device = device0)
        y[i] = func(x_tensor).detach().cpu().numpy()

    plt.plot(x, np.log10(y), color)

    if mark != None:
        val = func(mark).detach().cpu().numpy()
        npmark = mark.cpu().numpy()
        plt.plot(npmark, np.log10(val), 'bo')

    #plt.show()


fp_type = tc.float64

class polyfit():
    def __init__(self, X, Y, deg, find_best_deg = False):
        self.find_best_deg = find_best_deg
        assert deg > -1, "degree must be non-negative int"
        assert isinstance(deg, int), "degree must be non-negative int"
        self.deg = deg
        assert X.device == Y.device, "X and Y need to be on the same device"
        self.device = X.device

        p = tc.tensor([i for i in range(1, deg + 1)], device=self.device, dtype=fp_type)
        Xp = tc.full([deg + 1, len(X)], 1.0, device=self.device, dtype=fp_type)
        #print(tc.pow(X, p).t())
        Xp[1:(deg + 1)] = tc.pow(X.unsqueeze(1), p).t()
        #A = tc.linalg.lstsq(Xp.t(), Y).solution

        def poly_test(A, X, deg):  # Horner
            Y = A[-1]
            for i in range(1, deg + 1):
                Y = Y * X + A[-(1 + i)]
            return Y

        if self.find_best_deg:
            min_deg = 0
        else:
            min_deg = self.deg

        A_candidates = [None] * (deg + 1)
        mean_residuals = [None] * (deg + 1)
        #best_fit_mse = 2 ** 32
        best_fit_degree = 0
        for i in range(min_deg, deg + 1):
            Xpi = Xp[:(i + 1)]
            lstqfit = tc.linalg.lstsq(Xpi.t(), Y)
            Ai = lstqfit.solution
            A_candidates[i] = Ai
            mean_residuals[i] = tc.mean((poly_test(Ai, X, i) - Y) ** 2)
            if i == min_deg:
                best_fit_mse = mean_residuals[i]
                best_fit_degree = i
            if mean_residuals[i] < best_fit_mse:
                best_fit_degree = i
                best_fit_mse = mean_residuals[i]
        #print(best_fit_degree, "bfd")
        self.best_deg = best_fit_degree
        # best_fit_degree = argmin(mean_residuals)
        A = A_candidates[best_fit_degree]

        def polynomial(X):  # Horner
            Y = A[-1]
            if len(A) == 1:
                return 0 * X + Y
            for i in range(1, len(A)):
                Y = Y * X + A[-(1 + i)]
            return Y

        self.poly = polynomial

        if self.deg > self.best_deg:
            print(A)
            self.A = tc.full([self.deg + 1], 0.0, dtype =fp_type)
            self.A[:self.best_deg + 1] = A
        else:
            self.A = A

    def __call__(self, X):
        return self.poly(X)

# X = tc.tensor([0.0, 1.0, 2.0, 17.123])
# Y = 0.5 * (X ** 2) - 2 * X + 1/3
#
# poly = polyfit(X, Y, 8)
#
# print(Y)
# print(poly(X))
# print(poly.A)

