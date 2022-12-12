import numpy as np
import torch as tc
from typing import Union, Callable
import math

device0="cuda:0"
fp_type = tc.float

class asset():
    def __init__(self, S0: float, mu: Union[float, Callable], sigma: Union[float, Callable]):
        self.model = "Black-Scholes"
        self.S0 = S0

        self.mu = mu
        self.sigma = sigma

        self.has_local_volatility = callable(sigma)
        self.has_local_drift = callable(mu)

        self.list_of_params = [self.mu, self.sigma]

        self.has_any_local_params = any([callable(param) for param in self.list_of_params])
        self.has_any_global_params = not all([callable(param) for param in self.list_of_params])





class asset_collection():
    def __init__(self, assets, corr_matrix=None, risk_free=None):
        self.assets = assets
        if len(set([asset.model for asset in self.assets])) > 1:
            raise ValueError("assets in asset_collection need same market model")
        self.model = self.assets[0].model

        self.n_assets = len(assets)

        self.corr_matrix = corr_matrix
        if self.corr_matrix == None:
            self.corr_matrix = tc.eye(self.n_assets, dtype=fp_type, device=device0)
            self.corr_cholmat_L = self.corr_matrix
            self.corr_cholmat_R = self.corr_matrix
            self.correlated = False
        else:
            self.corr_matrix = tc.tensor(self.corr_matrix, dtype=fp_type, device=device0)
            self.corr_cholmat_L = tc.cholesky(self.corr_matrix)
            self.corr_cholmat_R = self.corr_cholmat_L.t()
            self.correlated = True


        self.S0 = [asset.S0 for asset in assets]
        self.Mu = [asset.mu for asset in assets]
        self.Sigma = [asset.sigma for asset in assets]
        self.mu_t = tc.tensor(self.Mu, device=device0)
        self.sigma_t = tc.tensor(self.Sigma, device=device0)

        self.has_local_drift = any([asset.has_local_drift for asset in self.assets])
        self.has_local_volatility = any([asset.has_local_volatility for asset in self.assets])
        self.has_local_params = self.has_local_drift or self.has_local_volatility

        self.has_any_local_params = any([asset.has_any_local_params for asset in self.assets])
        self.has_any_global_params = any([asset.has_any_global_params for asset in self.assets])

        #if self.has_any_local_params and self.has_any_global_params:
        #    self.make_all_local_params()

        if risk_free == None:
            self.risk_free = self.Mu[0]
        else:
            self.risk_free = risk_free

        #if self.model == "Heston":
        if False:
            self.V0 = [asset.v0 for asset in self.assets]
            self.Kappa = [asset.kappa for asset in self.assets]
            self.vTheta = [asset.vtheta for asset in self.assets]
            self.Rho = [asset.rho for asset in self.assets]
            self.sqrt_one_minus_Rhosq = np.sqrt(1 - np.square(self.Rho)).tolist()

    #def disc_step(self, dW):

    def Milstein_step(self, S_t0, dW_t, dt):
        #S_t1 = S_t0 * tc.exp((self.mu_t - 0.5 * (self.sigma_t ** 2)) * dt + self.sigma_t * dW_t)
        S_t1 = S_t0 + S_t0 * ( self.mu_t * dt + self.sigma_t * dW_t + 0.5 * ( self.sigma_t ** 2 ) * ((dW_t ** 2) - dt) )
        return S_t1

    def Euler_step(self, S_t0, dW_t, dt):
        S_t1 = S_t0 + S_t0 * (self.mu_t * dt + self.sigma_t * dW_t )
        return S_t1

    def get_mu_t(self, S_paths):
        return tc.tensor(self.Mu)

    def get_sigma_t(self, S_paths):
        return tc.tensor(self.Sigma)

    def disc_step(self):
        pass

    def make_all_local_params(self):
        for param_list in [self.Mu, self.Sigma]:
            print("test123123")
            for i in range(len(param_list)):
                if not callable(param_list[i]):
                    param_list[i] = make_float_callable(param_list[i])
                    print("float made callable")
                    #print(param(12423))

    def wrap_local_function(self, param):
        def tensor_param_func(*args, **kwargs):
            return tc.stack([param[i](*args[i], **kwargs) for i in range(self.n_assets)])
        return tensor_param_func

    def wrap_local_function(self, param):
        def tensor_param_func(s, *args, **kwargs):
            return tc.stack([ tc.full_like(s[0], param[i](s[i], *args, **kwargs))
                              if type(param[i](s[i], *args, **kwargs)) == float
                              else param[i](s[i], *args, **kwargs)
                              for i in range(self.n_assets)])

        return tensor_param_func


class asset_heston():
    def __init__(self, S0: float, mu: Union[float, Callable], sigma: Union[float, Callable], v0, kappa, vtheta, rho):
        self.model = "Heston"
        self.S0 = S0

        self.mu = mu
        self.sigma = sigma

        self.has_local_volatility = callable(sigma)
        self.has_local_drift = callable(mu)
        self.v0 = v0
        self.kappa = kappa
        self.vtheta = vtheta
        self.rho = rho

        self.list_of_params = [self.mu, self.sigma, self.kappa, self.vtheta, self.rho]

        self.has_any_local_params = any([callable(param) for param in self.list_of_params])
        self.has_any_global_params = not all([callable(param) for param in self.list_of_params])

class asset_collection_heston(asset_collection):
    def __init__(self, assets, corr_matrix=None, risk_free=None):
        super().__init__(assets, corr_matrix=None, risk_free=None)

        self.V0 = [asset.v0 for asset in self.assets]
        self.kappa = tc.tensor([asset.kappa for asset in self.assets], device=device0)
        self.vtheta = tc.tensor([asset.vtheta for asset in self.assets], device=device0)
        self.rho = tc.tensor([asset.rho for asset in self.assets], device=device0)
        self.sqrt_one_minus_rhosq = tc.tensor(tc.sqrt(1 - tc.square(self.rho)).tolist(), device=device0)

        #self.Milstein_step = Milstein_step_assets

    def Euler_step_assets(self, S_t0, dWs_t, v_t0, sqrt_v_t0, dt):
        S_t1 = S_t0 + S_t0 * (self.mu_t * dt + sqrt_v_t0 * dWs_t) \
               #+ 0.5 * S_t0 * (sqrt_v_t0 ** 2) * ((dWs_t ** 2) - dt)
        return S_t1

    def Euler_step_vola(self, v_t0, sqrt_v_t0, dWv_t, dt):
        v_t1 = v_t0 \
               + self.kappa * dt * (self.vtheta - v_t0) \
               + self.sigma_t * sqrt_v_t0 * dWv_t \
               #+ 0.25 * (self.sigma_t ** 2) * ((dWv_t ** 2) - dt)
        return v_t1

    def Milstein_step_assets(self, S_t0, dWs_t, v_t0, sqrt_v_t0, dt):

        S_t1 = S_t0 + S_t0 * (self.mu_t * dt  + sqrt_v_t0 * dWs_t ) \
                + 0.5 * S_t0 * (sqrt_v_t0 ** 2 ) * ((dWs_t ** 2) - dt)
        return S_t1



    def Milstein_step_vola(self, v_t0, sqrt_v_t0, dWv_t, dt):
        v_t1 = v_t0 \
               + self.kappa * dt * (self.vtheta - v_t0 ) \
               + self.sigma_t * sqrt_v_t0 * dWv_t \
               + 0.25 * (self.sigma_t ** 2) * ((dWv_t ** 2) - dt)
        return v_t1

    def Milstein_step_assets1(self, logS_t0, dWs_t0, dWv_t0, v_t0, sqrt_v_t0, dt):
        #S_t1 = S_t0 * tc.exp( (self.mu_t - 0.5 * v_t0) * dt + sqrt_v_t0 * dWs_t)
        logS_t1 = logS_t0 + (self.mu_t - 0.5 * v_t0) * dt  + sqrt_v_t0 * dWs_t0 + 0.25 * self.sigma_t * dWs_t * dWv_t0
        return logS_t1

    def Milstein_step_vola1(self, v_t0, sqrt_v_t0, dWv_t, dt):
        v_t1 = v_t0 \
               + self.kappa * dt * (self.vtheta - v_t0 ) \
               + self.sigma_t * sqrt_v_t0 * dWv_t \
               + 0.5 * (self.sigma_t ** 4) * ((dWv_t ** 2) - dt)
        return v_t1