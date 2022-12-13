


import math

import numpy as np
import torch
import torch as tc
import sys
from termcolor import colored
import matplotlib.pyplot as plt
from assets import *
from options import *
from paths import *
from integrators import *
from utils import *
#from mcalgo2tests import *
import time

from typing import Tuple

import os,time

clear = lambda: os.system('cls')

print(tc.__version__)

benchOP_standard = asset_collection([asset(90, 0.03, 0.15)])
asset_coll = benchOP_standard
#option_st_C = option(100, 1, benchOP_standard, payofftype="Call")
#option_st_C = vanilla_option(benchOP_standard, 100, 1, payofftype="Call")
#option1 = option_st_C



#fp_type = tc.float
fp_type = tc.float64
int_type = tc.long

class MultiLevelMonteCarlo():
    def __init__(self, option1, asset_coll, **kwargs):
        self.orig_kwargs = kwargs
        self.init_kwargs = kwargs.copy()
        self.option = option1
        self.asset_coll = asset_coll
        self.T = self.option.expiry
        #self.Integrator = Integrator(asset_coll)

        self.device = "cuda:0"
        self.M = 2 #4  #2
        self.Richardson_extrapol = True
        #self.M = kwargs.get("M", 2)
        self.n_steps_init = kwargs.get("n_steps_init", 1)
        self.n_steps_init = 1
        self.L_min = 4 #-self.M #- int(math.log(self.M, self.n_steps_init))
        self.L_max = 10
        self.fixed_L = kwargs.get("fixed_L", None)
        #self.fixed_L = 8

        if self.fixed_L != None:
            #fixed_L = 6
            self.L_min = self.fixed_L
            #self.L_min = 6 # -self.M
            self.L_max = self.L_min
        self.L = min(self.L_min, self.L_max)
        self.n_levels = self.L_max + 1
        #self.M = kwargs.get("M", 2)

        self.error_epsilon = kwargs.get("error_epsilon", 0.001)
        self.eps = self.error_epsilon
        self.epsilon_bias_split = kwargs.get("epsilon_bias_split", 0.5)
        self.eps_lambda = self.epsilon_bias_split
        self.epsilon_var_split = 1 - self.epsilon_bias_split

        self.level_factors = [(self.M ** l) for l in range(self.n_levels)]
        self.n_steps = [self.n_steps_init * self.level_factors[l] for l in range(self.n_levels)]
        self.n_steps_sub = [None] + [self.n_steps_init * self.level_factors[l - 1] for l in range(1, self.n_levels)]
        self.dt = [self.option.expiry / self.n_steps[l] for l in range(self.n_levels)]
        self.dt_sub = [None] + self.dt[:-1]
        self.sqrt_dt = np.sqrt(self.dt).tolist()
        self.sqrt_dt_sub = [None] + self.sqrt_dt[:-1]
        #print(self.dt_sub)

        self.time_sim = 0
        self.time_prep = 0
        self.time_total = 0
        self.level_time_prep = [0] * self.n_levels
        self.level_time_sum = [0] * self.n_levels
        self.level_time_running = [0] * self.n_levels
        self.level_batch_time = [0] * self.n_levels
        self.level_batch_time_running = [0] * self.n_levels

        self.ImportanceSampling = kwargs.get("ImportanceSampling", False)
        self.AntitheticSampling = kwargs.get("AntitheticSampling", False)
        self.max_training_level = 8 #self.L_max #8 #6
        #print("allocation Theta", self.asset_coll.n_assets * self.n_steps)
        self.Theta = [tc.tensor([[0.0] * self.asset_coll.n_assets] , device="cuda:0") for l in
                      range(self.n_levels)]
        #self.Theta = [tc.tensor([[i] * self.asset_coll.n_assets], device = "cuda:0")  for i in [1.370, 1.800, 2.0482, 2.155, 2.321, 2.314, 2.263, 2.279, 2.236]]
        self.dTheta = [tc.tensor([[0.0] * self.asset_coll.n_assets] , device="cuda:0") for l in
                      range(self.n_levels)]
        #self.Theta = [tc.tensor([[0.0] * self.asset_coll.n_assets] * self.n_steps[l], device="cuda:0") for l in range(self.n_levels)]
        #n_sims = 2 ** 12
        #self.n_sims_per_batch = 2 ** 13 #2 ** 13
        self.n_sims_per_batch = 2 ** 13 #2 ** 13
        self.n_sims_per_batch *= 2
        #if self.AntitheticSampling:
        #    self.n_sims_per_batch *= 2
        #self.n_sims_per_batch *= 8

        self.effective_n_sims_per_batch = tc.full([self.n_levels], self.n_sims_per_batch, dtype=int_type, device="cuda:0")
        if self.AntitheticSampling: # and False:
            self.effective_n_sims_per_batch[1:] = self.effective_n_sims_per_batch[1:] / 2
        #self.n_sims_per_batch = (2 ** 9) * 3 * 5    # 2 ** 13


        self.level_res = tc.full([self.n_levels], 0.0, dtype=fp_type, device="cuda:0")
        self.level_var = tc.full([self.n_levels], 0.0, dtype=fp_type, device="cuda:0")

        self.level_res_running = tc.full([self.n_levels], 0.0, dtype=fp_type, device="cuda:0")
        self.level_var_running = tc.full([self.n_levels], 0.0, dtype=fp_type, device="cuda:0")

        self.batch_res_sum = tc.full([self.n_levels], 0.0, dtype=fp_type, device="cuda:0")
        self.batch_var_sum = tc.full([self.n_levels], 0.0, dtype=fp_type, device="cuda:0")

        self.n_batches_done = tc.full([self.n_levels], 0, dtype=int_type, device="cuda:0")
        self.n_batches_req = [0] * self.n_levels
        self.n_sims_req = tc.full([self.n_levels], 0, dtype=int_type, device="cuda:0")
        #self.n_batches_req = tc.tensor([int(1) for l in range(self.n_levels)], dtype=int_type, device="cuda:0")
        self.n_batches_req = tc.tensor([int(0) for l in range(self.n_levels)], dtype=int_type, device="cuda:0")
        self.n_batches_req[:(self.L + 1)] = tc.tensor([int(1) for l in range(self.L + 1)], dtype=int_type, device="cuda:0")
        #self.n_batches_done = [0] * self.n_levels

        #convergence params
        self.alpha, self.beta = None, None

        self.set_integrator()

        #self.sync_cuda = False
        self.sync_cuda = True

        self.n_batches_discarded = tc.tensor([int(0) for l in range(self.n_levels)], dtype=int_type, device="cuda:0")
        self.total_C = 0

        self.testing_mode = False
        #self.testing_mode = True
        if self.testing_mode:
            self.L_min = 16
            self.L_max = 16
            self.L = self.L_min
            self.n_levels = self.L_max + 1
            self.n_batches_req = tc.tensor([int(math.ceil(2 ** 14 / (2 ** l))) for l in range(self.n_levels)], dtype=int_type,
                                           device="cuda:0")

    def set_integrator(self):
        if self.asset_coll.model == "Black-Scholes":
            self.Integrator = Integrator(self.asset_coll, M = self.M, ImportanceSampling = self.ImportanceSampling)
            self.S_paths = paths(self.n_sims_per_batch, self.asset_coll, 0, calling_mcalgo=self)
            self.option.prepare_paths(self.S_paths)
        elif self.asset_coll.model == "Heston":
            self.Integrator = Heston_Integrator(self.asset_coll, M = self.M, ImportanceSampling=self.ImportanceSampling)
            self.S_paths = paths_Heston(self.n_sims_per_batch, self.asset_coll, 0, calling_mcalgo=self)
            self.option.prepare_paths(self.S_paths)

    def reset(self):
        #("reset")
        self.__init__(self.option, self.asset_coll, **self.orig_kwargs)
        #print(self.level_res)
        #print(self.n_batches_done)
        #print(self.n_batches_req)

    def sim(self):
        #self.strata_optim_theta()
        return self.sim_mlmc()

    def sim_train_level(self, l ):
        print(l, "training level")
        train_timer_start0 = time.time()
        self.S_paths.reset_tensors(l, self)
        # if l == 0:
        #     plot_vars(self, l)
        #theta_l = self.adj_grid_lvl_up(self.Theta[l-1]) if l > 0 else self.Theta[l]
        theta_l = self.Theta[l-1] if l > 0 else self.Theta[l]
        #theta_l = tc.tensor([0.0], device = device0)
        #self.S_paths.theta = theta_l
        if True:
            self.S_paths.theta = theta_l
        else:
            self.S_paths.theta = tc.tensor([0.0], device = device0)
        print("sim train loop", theta_l)
        self.S_paths.ImportanceSampling_training = True
        self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l])

        self.S_paths.ImportanceSampling_training = False
        train_timer_start1 = time.time()
        dtheta, val = self.S_paths.optimize_theta(option= self.option, theta_init=theta_l)
        self.dTheta[l] = dtheta
        #self.Theta[l]
        oVar, E = self.S_paths.var_mean(option = self.option)
        if val >  1.1 * (oVar + E ** 2):
            self.dTheta[l] = -1000
            dtheta = 0
        #print("path var mean")
        #print(oVar, E)
        #print(dtheta)
        #print(oVar, "before")
        #print(self.Theta[l], "before", l)
        print(self.Theta[l], "before")
        if l < 0:
            self.Theta[l] = self.Theta[l-1] #if l > 0 else self.Theta[l]
        else:
            self.Theta[l] = theta_l + dtheta
        print(self.Theta[l], "after")
        if self.sync_cuda:
            tc.cuda.synchronize(device="cuda:0")
        train_timer_end = time.time()
        #if not (oVar > 1.1 * ( val - E ** 2)):
        if not (oVar > 1.051 * (val - E ** 2)):
            print("training paths used in calc")

            discard_all = True
            if not discard_all:
                self.batch_res_sum[l] = self.batch_res_sum[l] + E
                self.batch_var_sum[l] = self.batch_var_sum[l] + oVar
                self.n_batches_done[l] = self.n_batches_done[l] + 1
            else:
                self.n_batches_discarded[l] = self.n_batches_discarded[l] + 1
            self.level_time_prep[l] = train_timer_end - train_timer_start1
        else:
            print("training paths discarded")
            self.level_time_prep[l] = train_timer_end - train_timer_start0

        if l == self.max_training_level and False:
            for l1 in range(l + 1, self.L_max + 1):
                self.Theta[l1] = self.adj_grid_lvl_up(self.Theta[l1-1])
                #self.theta[]
            self.Integrator.ImportanceSampling_training = False
            self.S_paths.ImportanceSampling_training = False
            self.S_paths.dW = None #trim
            for l2 in range(self.L_max + 1):
                print(self.Theta[l2], "Theta", l2)


    def sim_level_test(self, l, n_sims = None,  IS_test = False):
        n_steps = self.n_steps[l]
        if n_sims == None:
            n_sims = self.n_sims_per_batch
        if not IS_test:
            ImportanceSampling = False
            theta_l = None
        else:
            ImportanceSampling = True
            theta_l = tc.tensor([[1] * self.asset_coll.n_assets], device="cuda:0")
        Integrator1 = Integrator(self.asset_coll, ImportanceSampling=ImportanceSampling)
        S_paths1 = paths(n_sims, self.asset_coll, 0, calling_mcalgo=self)
        S_paths1.reset_tensors(l, self)
        Integrator1.sim_paths_layered(S_paths1, self.option,
                                     n_steps, theta=theta_l)
        print(S_paths1.n_steps, "n_steps")
        print(S_paths1.n_steps_sub, "n_steps_sub")
        V1, E1 = S_paths1.var_mean(option=self.option)
        print(E1, V1)
        print( "dX")
        V2, E2 = S_paths1.var_mean_single(option=self.option)
        print(E2, V2)
        print( "Xfine")
        if l>0:
            V3, E3 = S_paths1.var_mean_single(option=self.option, sub = True)
            print(E3, V3)
            print("Xcoarse")

    def sim_level(self, l, n_batches):
        print(l, "sim level")
        theta_l = self.Theta[l] if self.ImportanceSampling else None
        #print(theta_l)
        timer_start = time.time()
        for j in range(n_batches):
            self.S_paths.reset_tensors(l, self)
            #self.S_paths = Integrator.sim_paths_layered(
            #print(self.option.expiry)
            self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option,
                                                             self.n_steps[l], theta=theta_l)
            batch_var, batch_res= self.S_paths.var_mean(option=self.option)
            print("level", l, j, "batch_var", batch_var)
            print("level", l, j , "batch_res", batch_res)
            self.batch_res_sum[l] = self.batch_res_sum[l] + batch_res
            self.batch_var_sum[l] = self.batch_var_sum[l] + batch_var
            if self.sync_cuda:
                tc.cuda.synchronize(device="cuda:0")
        timer_stop = time.time()
        #self.level_time_sum[l] += timer_stop - timer_start
        #self.level_time_avg_running[l] = self.level_time_sum[l] / n_batches
        self.level_batch_time[l] += timer_stop - timer_start
        self.level_batch_time_running[l] = self.level_batch_time[l] / self.n_batches_done[l]
        #self.level_batch_time[l]

        self.n_batches_done[l] = self.n_batches_done[l] + n_batches
        #print(level_res[l], level_var[l])

    def sim_mlmc(self):

        start = time.time()
        loop_counter = 0
        while any(self.n_batches_done[:(self.L + 1)] < self.n_batches_req[:(self.L + 1)]):
            print(self.n_batches_req)
            for l in range(self.L + 1):
                timer_start = time.time()
                if self.n_batches_done[l] < self.n_batches_req[l]:
                    print("level", l)
                    print(self.n_batches_req)
                    #if self.ImportanceSampling and not is_theta_optimized[l] and l <= self.max_training_level:
                    if self.ImportanceSampling and loop_counter < 1 and l <= self.max_training_level:
                        self.sim_train_level(l)
                    for j in range(int(self.n_batches_req[l] - self.n_batches_done[l]) ):
                        self.sim_level(l, 1)

                timer_end = time.time()
                self.level_time_sum[l] = self.level_time_sum[l] + (timer_end - timer_start)
            print("batch var sums", self.batch_var_sum[:(self.L + 1)] )
            self.level_var_running[:(self.L + 1)] = self.batch_var_sum[:(self.L + 1)] / self.n_batches_done[:(self.L + 1)]
            self.level_res_running[:(self.L + 1)] = self.batch_res_sum[:(self.L + 1)] / self.n_batches_done[:(self.L + 1)]
            print("Res", self.level_res_running[:(self.L + 1)])
            print("vars", self.level_var_running[:(self.L + 1)])
            if not self.testing_mode:
                self.calc_req_n_sims()
                if self.L > 4 and False:
                    self.estimate_convergence_params()
                #self.check_convergence()
                if not any(self.n_batches_done[:(self.L + 1)] < self.n_batches_req[:(self.L + 1)]):
                    self.check_convergence()
            print(self.n_batches_req)
            loop_counter += 1

        self.C = tc.sum( self.effective_n_sims_per_batch[:self.L+1] *
            tc.tensor(self.level_factors[:self.L+1], device=device0) * (self.n_batches_discarded[:self.L+1] + self.n_batches_done[:self.L+1]))

        self.level_var[:(self.L + 1)] = self.batch_var_sum[:(self.L + 1)] / self.n_batches_done[:(self.L + 1)]
        self.level_res[:(self.L + 1)] = self.batch_res_sum[:(self.L + 1)] / self.n_batches_done[:(self.L + 1)]
        end = time.time()
        print(end - start, "s", "asdasdasdad")


        self.time_total = sum(self.level_time_sum)

        Res = float(tc.sum(self.level_res[:(self.L + 1)], dim=0))
        Var = float(tc.sum(self.level_var[:(self.L + 1)], dim=0))

        if self.Richardson_extrapol:
            Res = Res + (1/(self.M - 1)) * float(self.level_res[self.L])
        r = asset_coll.risk_free
        T = self.option.expiry
        discount = math.exp(- r * T)
        #discount = 1
        return discount * Res, Var

    def check_convergence(self):
        if self.L_max == 0:
            return None
        n = min(3, self.L)
        #n = 1
        print("convegrence checkkkkkk")
        print(self.level_res_running[:(self.L + 1)][-n:])
        lastnE = self.level_res_running[:(self.L + 1)][-n:]
        lastnV = self.level_var_running[:(self.L + 1)][-n:]
        use_confidence_intervalls =  False
        #use_confidence_intervalls =  True
        if use_confidence_intervalls:
            std_factor = 0.5
            effective_n_sims_per_batch = self.n_sims_per_batch
            confidence_sigma = std_factor * tc.sqrt(lastnV / ( self.n_batches_done[:(self.L + 1)][-n:] * effective_n_sims_per_batch ))
            #("confidence", tc.flip(confidence))
            #print(tc.abs(tc.flip(lastnE, [0]) /  tc.tensor(self.level_factors[:n], device="cuda:0")))
            #print("confidence", tc.flip(confidence_sigma, [0]))
        else:
            confidence_sigma = 0
        print(tc.flip(tc.abs(lastnE) + confidence_sigma, [0]) / tc.tensor(self.level_factors[:n], device="cuda:0"))
        max_consec = tc.max(tc.flip(tc.abs(lastnE) + confidence_sigma, [0]) / tc.tensor(self.level_factors[:n], device="cuda:0"))
        #terminal_condition = (max_consec < (self.M - 1) * self.eps / math.sqrt(2))
        terminal_condition = (max_consec < (self.M - 1) * self.eps / math.sqrt( 1/self.epsilon_bias_split ))


        if self.Richardson_extrapol:
            canary = abs(float(self.level_res_running[(self.L)])
                                     - (1/self.M) * float(self.level_res_running[(self.L - 1)]))
            canary2 = abs(float(self.level_res_running[(self.L - 1)])
                         - (1 / self.M) * float(self.level_res_running[(self.L - 2)]))
            terminal_condition = max(canary, canary2) < float((self.M ** 2 - 1) * self.eps / math.sqrt( 1/self.epsilon_bias_split ) )

        if terminal_condition:
            print("convergence detected ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß", self.L)
        else:
            print("no convergence, more levels req")
            if self.L < self.L_max:
                self.L = self.L + 1
                self.n_batches_req[self.L] = self.n_batches_req[self.L] + 1

                #self.n_levels = self.n_levels + 1


    def estimate_convergence_params(self):
        l0 = 1
        #dt = tc.tensor(self.dt[:self.L +1], device= device0, dtype =fp_type)
        #logdt = tc.log(dt)
        ls = tc.tensor([l for l in range(l0,self.L +1)], device= device0, dtype =fp_type)
        dts = tc.tensor(self.dt[l0:self.L + 1], device= device0, dtype =fp_type)
        V = tc.abs(self.level_var_running[l0:self.L +1])
        V[1:] = tc.maximum(0.5 * V[:-1], V[1:])
        logV = tc.log(tc.abs(V))
        E = tc.abs(self.level_res_running[l0:self.L +1])
        E[1:] = tc.maximum(0.5 * E[:-1], E[1:])
        logE = tc.log(tc.abs(E))

        #linear_fitV = polyfit(ls, logV, 1)
        linear_fitV = polyfit(tc.log(dts), logV, 1)
        #print(self.L - l0, "data points")
        #print(linear_fitV.A, "V poly A")
        self.beta = float(linear_fitV.A[1])
        #print(V, "V")
        #print(logV, "logV")
        #linear_fitE = polyfit(ls, logE, 1)
        linear_fitE = polyfit(tc.log(dts), logE, 1)
        #print(linear_fitE.A, "E poly A")
        self.alpha = float(linear_fitE.A[1])
        #print(E, "E")
        #print(logE, "logE")


    def calc_req_n_sims(self):
        """Calculate the required number of simulations per level for 0.5 * eps ** 2 total variance """
        V = tc.tensor(self.level_var_running[:(self.L + 1)], device="cuda:0")
        h = tc.tensor(self.dt[:(self.L + 1)], device="cuda:0")
        print("V", V)
        print("h", h)
        sqrt_V_div_dt = tc.sqrt(tc.div(V, h))
        sums_sqrt_V_div_dt = tc.cumsum(sqrt_V_div_dt, 0)
        sums_sqrt_V_div_dt = tc.sum(sqrt_V_div_dt, 0)
        #n_sims_req = tc.ceil(2 * (self.eps ** - 2) * (tc.sqrt(V * h)) * sums_sqrt_V_div_dt)
        n_sims_req = tc.ceil( (1/self.epsilon_var_split ) * (self.eps ** -2) * (tc.sqrt(V * h)) * sums_sqrt_V_div_dt)
        print("reqqqq n_sims", n_sims_req)

        n_batches_req = tc.ceil(n_sims_req / self.effective_n_sims_per_batch[:(self.L + 1)])
        print("reqqqq n_sims", n_batches_req)
        self.n_batches_req[:(self.L + 1)] = n_batches_req
        print(self.n_batches_req.dtype)

    def adj_grid_lvl_up(self, inp):
        n_s, n_a = inp.shape # n_steps * n_assets
        out = tc.empty([n_s * self.M, n_a], device = self.device)
        for i in range(n_s):
            out[i * self.M: (i + 1) * self.M] = inp[i].repeat(self.M, 1)
        return out

    def adj_grid_lvl_down(self, inp):
        n_s, n_a = inp.shape
        out = tc.empty([int(n_s / self.M), n_a], device = self.device)
        for i in range(int(n_s / self.M)):
            out[i] = tc.mean(inp[i * self.M: (i + 1) * self.M])
        return out

