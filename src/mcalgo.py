


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

        #for l in range(self.L + 1):
        #    self.level_var[l] = self.batch_var_sum[l] / self.n_batches_done[l]
        #    self.level_res[l] =  self.batch_res_sum[l] / self.n_batches_done[l]
        self.C = tc.sum( self.effective_n_sims_per_batch[:self.L+1] *
            tc.tensor(self.level_factors[:self.L+1], device=device0) * (self.n_batches_discarded[:self.L+1] + self.n_batches_done[:self.L+1]))

        self.level_var[:(self.L + 1)] = self.batch_var_sum[:(self.L + 1)] / self.n_batches_done[:(self.L + 1)]
        self.level_res[:(self.L + 1)] = self.batch_res_sum[:(self.L + 1)] / self.n_batches_done[:(self.L + 1)]
        end = time.time()
        print(end - start, "s", "asdasdasdad")

        #print(self.Theta[:4])
        for l in range(self.L + 1):
            pass
            #self.level_batch_avg_time[l] = self.level_batch_time[l] / int(self.n_batches_done[l])
        self.time_total = sum(self.level_time_sum)

        Res = float(tc.sum(self.level_res[:(self.L + 1)], dim=0))
        Var = float(tc.sum(self.level_var[:(self.L + 1)], dim=0))
        #Richardson_extrapol = True
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



class MonteCarlo(MultiLevelMonteCarlo):
    def __init__(self, option1, asset_coll, n_steps, **kwargs):

        kwargs["fixed_L"] = 0
        kwargs["epsilon_bias_split"] = 0
        super().__init__(option1, asset_coll, kwargs)

    def calc_req_n_sims(self):
        V = tc.tensor(self.level_var_running[0], device="cuda:0")



if False:
    def strata_optim_theta(self):

        theta_base = 1.5
        #n_theta_steps = 3
        #theta_step_size = [2, 1, 1]
        #n_theta_steps = 1
        #theta_step_size = [1]
        n_theta_steps = 2
        theta_step_size = [1, 1]
        #theta_step_size = [1, 3]
        sum_stepsizes = sum(theta_step_size)
        M = 4
        n_strata = M ** (n_theta_steps - 1)
        dXsq_stratas = [None] * n_strata
        dX_stratas = [None] * n_strata
        dW_stratas = [None] * n_strata
        l = 2
        #l = 0
        #l = 1

        n_ass = 1
        n_steps = self.n_steps[l]
        T = 1
        n_dW_theta_steps = [int((theta_step_size[i] / sum_stepsizes) * n_steps) for i in range(n_theta_steps) ]
        print(n_steps, n_dW_theta_steps)
        dt =  1 /n_steps
        dT = [[step_size * dt] * n_ass for step_size in n_dW_theta_steps]
        print("n strata", n_strata)
        #self.S_paths.reset_tensors(l, self)
        #self.S_paths.ImportanceSampling_training = True
        #self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
        #                                                 theta=None)
        for i in range(n_strata ):
            print("strata#", i)
            self.S_paths.reset_tensors(l, self)
            #self.S_paths.stratum_i = None #i
            self.S_paths.stratum_i = i
            theta_l = tc.tensor([theta_base] * n_steps, device=device0)
            self.S_paths.ImportanceSampling_training = True
            self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                             theta=theta_l)
            dXsq_stratas[i] = self.S_paths.dX() ** 2
            dX_stratas[i] = tc.var_mean(self.S_paths.dX())
            print(i, tc.mean(dXsq_stratas[i]), "strata sq mean")
            print(i)
            dW = self.S_paths.dW.clone()
            print("dW shape ", dW.shape)
            shape = list(dW.shape)
            #n_steps = shape[0]
            dW_stratas[i] = tc.empty([n_theta_steps] + shape[1:], device = device0)
            li = 0
            for j in range(n_theta_steps ):
                ri = li + n_dW_theta_steps[j]
                print("l, r", li, ri)
                print("dW shape ",dW.shape)
                dW_stratas[i][j] = tc.sum(dW[li:ri ], dim = 0, keepdim= False)
                print("shape", dW_stratas[i][j].shape)
                print("var mean", tc.var_mean(dW_stratas[i][j]))
                li = ri
        self.S_paths.ImportanceSampling_training = False
        print("arrived $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        n_ass = 1
        theta_init = tc.tensor([[theta_base] * n_ass for i in range(int((M ** (n_theta_steps + 0) - 1) / (M - 1)))],
                               device=device0, requires_grad=True)
        print(theta_init, "thetainit")
        print(theta_init.shape, "shape thetainit")
        p_strata = 1 / n_strata
        n_ass = 1
        dT = tc.tensor(dT, device=device0)

        def EdXsqZ_Tensor(theta):
            #EdXsqZ = 0
            #step_theta = [None] * n_theta_steps
            step_theta = tc.full([n_theta_steps, n_strata, n_ass], 0.0, device=device0)
            for j in range(n_theta_steps):
                l = int((M ** (j + 0) - 1) / (M - 1))
                r = int((M ** (j + 1) - 1) / (M - 1) - 1)
                #print(l, r)
                theta_i0 = theta[l:r + 1].repeat_interleave(M ** (n_theta_steps - j - 1), dim=0)
                #print(theta_i0)
                step_theta[j] = step_theta[j] + theta_i0
            strata_theta = torch.transpose(step_theta, 0, 1)
            EdXsqZ = 0
            for i in range(n_strata):
                theta_i = strata_theta[i]
                integ_theta_dW = tc.sum(theta_i[:, None, :] * dW_stratas[i], dim=0)
                integ_thetasq_dt = tc.sum((theta_i ** 2) * dT)
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                EdXsqZ = EdXsqZ + p_strata * tc.mean(dXsq_stratas[i] * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
            #EdXsqZ.backward()
            return EdXsqZ #.data, theta.grad.data

        def EdXsqZ_TensorGrad(theta):
            #EdXsqZ = 0
            step_theta = [None] * n_theta_steps
            step_theta = tc.full([n_theta_steps, n_strata, n_ass], 0.0, device=device0)
            for j in range(n_theta_steps):
                l = int((M ** (j + 0) - 1) / (M - 1))
                r = int((M ** (j + 1) - 1) / (M - 1) - 1)
                #print(l, r, "l r tree")
                theta_i0 = theta[l:r + 1].repeat_interleave(M ** (n_theta_steps - j - 1), dim=0)
                #print(theta_i0, "theta i0")
                step_theta[j] = step_theta[j] + theta_i0
            strata_theta = torch.transpose(step_theta, 0, 1)
            EdXsqZ = 0
            for i in range(n_strata):
                theta_i = strata_theta[i]
                print(theta_i, theta_i.shape, "thet_i + shape")
                print(dW_stratas[i].shape, "stratas i shape")
                for dW in dW_stratas[i]:
                    print(tc.var_mean(dW), "stratas i dW var mean")
                integ_theta_dW = tc.sum(theta_i[:, None, :] * dW_stratas[i], dim=0)
                print(tc.var_mean(integ_theta_dW), "var_mean integthetadW")
                print(theta_i, "theta_i")
                print(dT, "dT")
                print((theta_i ** 2) * dT, "theta_i sq dT")
                integ_thetasq_dt = tc.sum((theta_i ** 2) * dT)
                print(integ_thetasq_dt, "integ_thetasq")
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                print(exp_integs.shape, "shape exp integs")
                print(tc.mean(exp_integs), "exp_integs mean")
                EdXsqZ = EdXsqZ + (p_strata ** 2) * tc.mean(dXsq_stratas[i] * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
            #theta.retain_grad()
            #EdXsqZ.backward()
            #return EdXsqZ.data, theta.grad.data
            return EdXsqZ

        def grad_desc(func, arg_init, max_steps = 100, lr = 1):
            i = 0
            val0 = func(arg_init)
            arg = arg_init
            arg.requires_grad = True

            factor = 10000
            def func1(arg):
                return factor * func(arg)

            while i < max_steps:
                print("grad desc", i, "///////////////////////////")
                #val, grad = func(arg)
                val = func1(arg)
                val.backward()
                grad = arg.grad
                grad = grad / tc.norm(grad)
                with tc.no_grad():
                    arg = arg - lr * grad
                arg.requires_grad = True
                i = i + 1
            val1 = func(arg)
            print(arg_init, "arginit")
            print(val0, "old val")
            print(val1, "new val")
            print(arg, "new arg")

            return arg

        theta_init = tc.tensor([[0.0] * n_ass for i in range(int((M ** (n_theta_steps + 0) - 1) / (M - 1)))],
                               device=device0, requires_grad=True)
        new_theta = grad_desc(EdXsqZ_TensorGrad, theta_init)
        print(grad_desc(EdXsqZ_TensorGrad, theta_init))
        print(dX_stratas)
        #print(EdXsqZ_TensorGrad(theta_init))
        mean, var  = 0, 0
        dX_stratas = [None] * n_strata
        n_tries = 1000
        new_theta.requires_grad = False
        #new_theta[1] = new_theta[1] * 0.3
        with HiddenPrints():
            theta_ls = [theta_base  + tc.tensor([float(new_theta[0]), float(new_theta[0]), float(new_theta[1+i]), float(new_theta[1+i])],device=device0) for i in range(n_strata)]
            for j in range(n_tries):
                dX_stratas = [None] * n_strata
                for i in range(n_strata ):
                    print("strata#", i)
                    self.S_paths.reset_tensors(l, self)
                    self.S_paths.stratum_i = i
                    #self.S_paths.stratum_i = i
                    #x123 =  tc.tensor([0.0, 0.0, -2.0, -2.3], device=device0)
                    theta_l = theta_ls[i]
                    #theta_l = 1.5 + tc.tensor([0.0, 0.0, 0.0, 0.0], device=device0)
                    # theta_l = 1.5 + tc.tensor([0.9, 1.0, 1.0, 2.2], device=device0)
                    #print(theta_l, "thetal strata", i)
                    self.S_paths.ImportanceSampling_training = True
                    self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                                     theta=theta_l)
                    dX_stratas[i] = tc.var_mean(self.S_paths.dX())
                mean = mean + p_strata  * sum([dX_stratas[i][1] for i in range(n_strata) ])
                var  = var  + (p_strata ** 2) * sum([dX_stratas[i][0] for i in range(n_strata) ])
            mean = mean/ n_tries
            var = var / n_tries
        for i in range(n_strata):
            pass
            #print(i, "theta ls", theta_l[i])
        print(theta_l )
        print("means", [float(x[1]) for x in dX_stratas])
        print(mean, "mean")
        print("vars", [float(x[0]) for x in dX_stratas])
        print(var, "var")
        with HiddenPrints():
            for j in range(n_tries):
                dX_stratas = [None] * n_strata
                for i in range(n_strata ):
                    print("strata#", i)
                    self.S_paths.reset_tensors(l, self)
                    self.S_paths.stratum_i = i
                    #self.S_paths.stratum_i = i
                    #x123 =  tc.tensor([0.0, 0.0, -2.0, -2.3], device=device0)
                    #theta_l = 0.0  + tc.tensor([new_theta[0], new_theta[0], new_theta[1+i], new_theta[1+i]], device=device0)
                    theta_l = theta_base + tc.tensor([0.0, 0.0, 0.0, 0.0], device=device0)
                    # theta_l = 1.5 + tc.tensor([0.9, 1.0, 1.0, 2.2], device=device0)
                    print(theta_l, "thetal strata", i)
                    self.S_paths.ImportanceSampling_training = True
                    self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                                     theta=theta_l)
                    dX_stratas[i] = tc.var_mean(self.S_paths.dX())
                mean = mean + p_strata * sum([dX_stratas[i][1] for i in range(n_strata)])
                var = var + (p_strata ** 2) * sum([dX_stratas[i][0] for i in range(n_strata)])
            mean = mean/ n_tries
            var = var / n_tries
        print("means", [float(x[1]) for x in dX_stratas])
        print(mean, "mean")
        print("vars", [float(x[0]) for x in dX_stratas])
        print(var, "var")
        quit()
        theta0 = tc.tensor([[0.5] * n_ass for i in range(int((M ** (n_theta_steps + 0) - 1) / (M - 1)))],
                               device=device0, requires_grad=False)
        theta1 = theta0 * 3
        print(EdXsqZ_TensorGrad(theta0))
        print( EdXsqZ_TensorGrad(theta1))

        for i in range(n_strata ):
            print("strata#", i)
            self.S_paths.reset_tensors(l, self)
            #self.S_paths.stratum_i = None #i
            self.S_paths.stratum_i = i

            self.S_paths.ImportanceSampling_training = True
            self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                             theta=None)

        quit()
        #print(EdXsqZ_TensorGrad(theta_init))
        #val0 = EdXsqZ_Tensor(theta_init)
        print("start GD it")
        start = time.time()
        theta = theta_init
        val, grad = EdXsqZ_TensorGrad(theta)
        #val, grad = EdXsqZ_TensorGrad(theta_init)

        grad = grad / tc.norm(grad)
        theta = theta_init
        theta.requires_grad = True

        lr = 1
        max_steps = 50 #10
        max_overshots = 100 #30
        steps = 0
        # while (not converged) and (steps < max_steps):
        overshot_counter = 0
        while (steps < max_steps):
            proposed_theta = theta - lr * grad
            theta = proposed_theta
            #theta.grad.zero_()
            val, grad = EdXsqZ_TensorGrad(theta)
            quit()
            if proposed_val < 1 * val:
                theta = proposed_theta
                # self.teta_path.append(teta)
                val = proposed_val

            grad = grad / tc.norm(grad)

        end = time.time()
        print("GRADDDIENTENVERFAHREN", end - start, "s")
        print(theta)
        print(val, "new val")
        print(val0, "old val")

        quit()


        while (steps < max_steps) and (overshot_counter < max_overshots):
            # print(teta)
            proposed_theta = theta - lr * grad
            #theta.grad.zero_()
            print(proposed_theta," prop theta")
            #proposed_val = EdXsqZ_Tensor(proposed_theta)
            proposed_val = EdXsqZ_TensorGrad(proposed_theta)[0]
            #with tc.no_grad():
            #    proposed_val = EdXsqZ_Tensor(proposed_theta)

            # print(proposed_val)
            if proposed_val < 0.95 * val:
                theta = proposed_theta
                #theta.grad.zero_()
                # self.teta_path.append(teta)
                val = proposed_val
            elif False and overshot_counter < max_overshots:
                lr = lr / 2
                overshot_counter += 1
                print(theta, "problemeatic theta")
                theta.requires_grad = True
                print("getting grads")
                val, grad = EdXsqZ_TensorGrad(theta)
                grad = grad / tc.norm(grad)
                theta.requires_grad = False

            steps += 1

        end = time.time()
        print("GRADDDIENTENVERFAHREN", end - start, "s")
        print(theta)
        print(val, "new val")
        print(val0, "old val")

        quit()



    def strata_optim_theta1(self):

        theta_base = 1.5
        #n_theta_steps = 3
        #theta_step_size = [2, 1, 1]
        #n_theta_steps = 1
        #theta_step_size = [1]
        n_theta_steps = 2
        theta_step_size = [1, 1]
        #theta_step_size = [1, 3]
        sum_stepsizes = sum(theta_step_size)
        M = 4
        n_strata = M ** (n_theta_steps - 1)
        dXsq_stratas = [None] * n_strata
        dX_stratas = [None] * n_strata
        dW_stratas = [None] * n_strata
        l = 2
        #l = 0
        #l = 1

        n_ass = 1
        n_steps = self.n_steps[l]
        T = 1
        n_dW_theta_steps = [int((theta_step_size[i] / sum_stepsizes) * n_steps) for i in range(n_theta_steps) ]
        print(n_steps, n_dW_theta_steps)
        dt =  1 /n_steps
        dT = [[step_size * dt] * n_ass for step_size in n_dW_theta_steps]
        print("n strata", n_strata)
        #self.S_paths.reset_tensors(l, self)
        #self.S_paths.ImportanceSampling_training = True
        #self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
        #                                                 theta=None)
        for i in range(n_strata ):
            print("strata#", i)
            self.S_paths.reset_tensors(l, self)
            #self.S_paths.stratum_i = None #i
            self.S_paths.stratum_i = i
            theta_l = tc.tensor([theta_base] * n_steps, device=device0)
            self.S_paths.ImportanceSampling_training = True
            self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                             theta=theta_l)
            dXsq_stratas[i] = self.S_paths.dX() ** 2
            dX_stratas[i] = tc.var_mean(self.S_paths.dX())
            print(i, tc.mean(dXsq_stratas[i]), "strata sq mean")
            print(i)
            dW = self.S_paths.dW.clone()
            print("dW shape ", dW.shape)
            shape = list(dW.shape)
            #n_steps = shape[0]
            dW_stratas[i] = tc.empty([n_theta_steps] + shape[1:], device = device0)
            li = 0
            for j in range(n_theta_steps ):
                ri = li + n_dW_theta_steps[j]
                print("l, r", li, ri)
                print("dW shape ",dW.shape)
                dW_stratas[i][j] = tc.sum(dW[li:ri ], dim = 0, keepdim= False)
                print("shape", dW_stratas[i][j].shape)
                print("var mean", tc.var_mean(dW_stratas[i][j]))
                li = ri
        self.S_paths.ImportanceSampling_training = False
        print("arrived $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        n_ass = 1
        theta_init = tc.tensor([[theta_base] * n_ass for i in range(int((M ** (n_theta_steps + 0) - 1) / (M - 1)))],
                               device=device0, requires_grad=True)
        print(theta_init, "thetainit")
        print(theta_init.shape, "shape thetainit")
        p_strata = 1 / n_strata
        n_ass = 1
        dT = tc.tensor(dT, device=device0)

        def EdXsqZ_Tensor(theta):
            #EdXsqZ = 0
            #step_theta = [None] * n_theta_steps
            step_theta = tc.full([n_theta_steps, n_strata, n_ass], 0.0, device=device0)
            for j in range(n_theta_steps):
                l = int((M ** (j + 0) - 1) / (M - 1))
                r = int((M ** (j + 1) - 1) / (M - 1) - 1)
                #print(l, r)
                theta_i0 = theta[l:r + 1].repeat_interleave(M ** (n_theta_steps - j - 1), dim=0)
                #print(theta_i0)
                step_theta[j] = step_theta[j] + theta_i0
            strata_theta = torch.transpose(step_theta, 0, 1)
            EdXsqZ = 0
            for i in range(n_strata):
                theta_i = strata_theta[i]
                integ_theta_dW = tc.sum(theta_i[:, None, :] * dW_stratas[i], dim=0)
                integ_thetasq_dt = tc.sum((theta_i ** 2) * dT)
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                EdXsqZ = EdXsqZ + p_strata * tc.mean(dXsq_stratas[i] * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
            #EdXsqZ.backward()
            return EdXsqZ #.data, theta.grad.data

        def EdXsqZ_TensorGrad(theta):
            #EdXsqZ = 0
            step_theta = [None] * n_theta_steps
            step_theta = tc.full([n_theta_steps, n_strata, n_ass], 0.0, device=device0)
            for j in range(n_theta_steps):
                l = int((M ** (j + 0) - 1) / (M - 1))
                r = int((M ** (j + 1) - 1) / (M - 1) - 1)
                #print(l, r, "l r tree")
                theta_i0 = theta[l:r + 1].repeat_interleave(M ** (n_theta_steps - j - 1), dim=0)
                #print(theta_i0, "theta i0")
                step_theta[j] = step_theta[j] + theta_i0
            strata_theta = torch.transpose(step_theta, 0, 1)
            EdXsqZ = 0
            for i in range(n_strata):
                theta_i = strata_theta[i]
                print(theta_i, theta_i.shape, "thet_i + shape")
                print(dW_stratas[i].shape, "stratas i shape")
                for dW in dW_stratas[i]:
                    print(tc.var_mean(dW), "stratas i dW var mean")
                integ_theta_dW = tc.sum(theta_i[:, None, :] * dW_stratas[i], dim=0)
                print(tc.var_mean(integ_theta_dW), "var_mean integthetadW")
                print(theta_i, "theta_i")
                print(dT, "dT")
                print((theta_i ** 2) * dT, "theta_i sq dT")
                integ_thetasq_dt = tc.sum((theta_i ** 2) * dT)
                print(integ_thetasq_dt, "integ_thetasq")
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                print(exp_integs.shape, "shape exp integs")
                print(tc.mean(exp_integs), "exp_integs mean")
                EdXsqZ = EdXsqZ + (p_strata ** 2) * tc.mean(dXsq_stratas[i] * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
            #theta.retain_grad()
            #EdXsqZ.backward()
            #return EdXsqZ.data, theta.grad.data
            return EdXsqZ

        def grad_desc(func, arg_init, max_steps = 100, lr = 1):
            i = 0
            val0 = func(arg_init)
            arg = arg_init
            arg.requires_grad = True

            factor = 10000
            def func1(arg):
                return factor * func(arg)

            while i < max_steps:
                print("grad desc", i, "///////////////////////////")
                #val, grad = func(arg)
                val = func1(arg)
                val.backward()
                grad = arg.grad
                grad = grad / tc.norm(grad)
                with tc.no_grad():
                    arg = arg - lr * grad
                arg.requires_grad = True
                i = i + 1
            val1 = func(arg)
            print(arg_init, "arginit")
            print(val0, "old val")
            print(val1, "new val")
            print(arg, "new arg")

            return arg

        theta_init = tc.tensor([[0.0] * n_ass for i in range(int((M ** (n_theta_steps + 0) - 1) / (M - 1)))],
                               device=device0, requires_grad=True)
        new_theta = grad_desc(EdXsqZ_TensorGrad, theta_init)
        print(grad_desc(EdXsqZ_TensorGrad, theta_init))
        print(dX_stratas)
        #print(EdXsqZ_TensorGrad(theta_init))
        mean, var  = 0, 0
        dX_stratas = [None] * n_strata
        n_tries = 1000
        new_theta.requires_grad = False
        #new_theta[1] = new_theta[1] * 0.3
        with HiddenPrints():
            theta_ls = [theta_base  + tc.tensor([float(new_theta[0]), float(new_theta[0]), float(new_theta[1+i]), float(new_theta[1+i])],device=device0) for i in range(n_strata)]
            for j in range(n_tries):
                dX_stratas = [None] * n_strata
                for i in range(n_strata ):
                    print("strata#", i)
                    self.S_paths.reset_tensors(l, self)
                    self.S_paths.stratum_i = i
                    #self.S_paths.stratum_i = i
                    #x123 =  tc.tensor([0.0, 0.0, -2.0, -2.3], device=device0)
                    theta_l = theta_ls[i]
                    #theta_l = 1.5 + tc.tensor([0.0, 0.0, 0.0, 0.0], device=device0)
                    # theta_l = 1.5 + tc.tensor([0.9, 1.0, 1.0, 2.2], device=device0)
                    #print(theta_l, "thetal strata", i)
                    self.S_paths.ImportanceSampling_training = True
                    self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                                     theta=theta_l)
                    dX_stratas[i] = tc.var_mean(self.S_paths.dX())
                mean = mean + p_strata  * sum([dX_stratas[i][1] for i in range(n_strata) ])
                var  = var  + (p_strata ** 2) * sum([dX_stratas[i][0] for i in range(n_strata) ])
            mean = mean/ n_tries
            var = var / n_tries
        for i in range(n_strata):
            pass
            #print(i, "theta ls", theta_l[i])
        print(theta_l )
        print("means", [float(x[1]) for x in dX_stratas])
        print(mean, "mean")
        print("vars", [float(x[0]) for x in dX_stratas])
        print(var, "var")
        with HiddenPrints():
            for j in range(n_tries):
                dX_stratas = [None] * n_strata
                for i in range(n_strata ):
                    print("strata#", i)
                    self.S_paths.reset_tensors(l, self)
                    self.S_paths.stratum_i = i
                    #self.S_paths.stratum_i = i
                    #x123 =  tc.tensor([0.0, 0.0, -2.0, -2.3], device=device0)
                    #theta_l = 0.0  + tc.tensor([new_theta[0], new_theta[0], new_theta[1+i], new_theta[1+i]], device=device0)
                    theta_l = theta_base + tc.tensor([0.0, 0.0, 0.0, 0.0], device=device0)
                    # theta_l = 1.5 + tc.tensor([0.9, 1.0, 1.0, 2.2], device=device0)
                    print(theta_l, "thetal strata", i)
                    self.S_paths.ImportanceSampling_training = True
                    self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                                     theta=theta_l)
                    dX_stratas[i] = tc.var_mean(self.S_paths.dX())
                mean = mean + p_strata * sum([dX_stratas[i][1] for i in range(n_strata)])
                var = var + (p_strata ** 2) * sum([dX_stratas[i][0] for i in range(n_strata)])
            mean = mean/ n_tries
            var = var / n_tries
        print("means", [float(x[1]) for x in dX_stratas])
        print(mean, "mean")
        print("vars", [float(x[0]) for x in dX_stratas])
        print(var, "var")
        quit()
        theta0 = tc.tensor([[0.5] * n_ass for i in range(int((M ** (n_theta_steps + 0) - 1) / (M - 1)))],
                               device=device0, requires_grad=False)
        theta1 = theta0 * 3
        print(EdXsqZ_TensorGrad(theta0))
        print( EdXsqZ_TensorGrad(theta1))

        for i in range(n_strata ):
            print("strata#", i)
            self.S_paths.reset_tensors(l, self)
            #self.S_paths.stratum_i = None #i
            self.S_paths.stratum_i = i

            self.S_paths.ImportanceSampling_training = True
            self.S_paths = self.Integrator.sim_paths_layered(self.S_paths, self.option, self.n_steps[l],
                                                             theta=None)

        quit()
        #print(EdXsqZ_TensorGrad(theta_init))
        #val0 = EdXsqZ_Tensor(theta_init)
        print("start GD it")
        start = time.time()
        theta = theta_init
        val, grad = EdXsqZ_TensorGrad(theta)
        #val, grad = EdXsqZ_TensorGrad(theta_init)

        grad = grad / tc.norm(grad)
        theta = theta_init
        theta.requires_grad = True

        lr = 1
        max_steps = 50 #10
        max_overshots = 100 #30
        steps = 0
        # while (not converged) and (steps < max_steps):
        overshot_counter = 0
        while (steps < max_steps):
            proposed_theta = theta - lr * grad
            theta = proposed_theta
            #theta.grad.zero_()
            val, grad = EdXsqZ_TensorGrad(theta)
            quit()
            if proposed_val < 1 * val:
                theta = proposed_theta
                # self.teta_path.append(teta)
                val = proposed_val

            grad = grad / tc.norm(grad)

        end = time.time()
        print("GRADDDIENTENVERFAHREN", end - start, "s")
        print(theta)
        print(val, "new val")
        print(val0, "old val")

        quit()


        while (steps < max_steps) and (overshot_counter < max_overshots):
            # print(teta)
            proposed_theta = theta - lr * grad
            #theta.grad.zero_()
            print(proposed_theta," prop theta")
            #proposed_val = EdXsqZ_Tensor(proposed_theta)
            proposed_val = EdXsqZ_TensorGrad(proposed_theta)[0]
            #with tc.no_grad():
            #    proposed_val = EdXsqZ_Tensor(proposed_theta)

            # print(proposed_val)
            if proposed_val < 0.95 * val:
                theta = proposed_theta
                #theta.grad.zero_()
                # self.teta_path.append(teta)
                val = proposed_val
            elif False and overshot_counter < max_overshots:
                lr = lr / 2
                overshot_counter += 1
                print(theta, "problemeatic theta")
                theta.requires_grad = True
                print("getting grads")
                val, grad = EdXsqZ_TensorGrad(theta)
                grad = grad / tc.norm(grad)
                theta.requires_grad = False

            steps += 1

        end = time.time()
        print("GRADDDIENTENVERFAHREN", end - start, "s")
        print(theta)
        print(val, "new val")
        print(val0, "old val")

        quit()

        def optimize_theta(self, dW_stratas, dXsq_stratas, theta_init= theta_init):
            start = time.time()
            #dX = self.dX(option=option)

            #dXsq = dX ** 2
            T = option.expiry
            n_steps = self.dW.shape[0]
            dt = T / n_steps

            #constant_theta = self.antithetic
            # constant_theta = False
            constant_theta = True
            #if theta_init == None:
            #    if constant_theta:
            #        theta_init = tc.tensor([0.0] * self.n_assets, device=device0, requires_grad=True)
            #    else:
            #        theta_init = tc.full([n_steps, self.n_assets], 0.0, device=device0, requires_grad=True)

            p_strata = 1 / n_strata
            n_ass = 1
            dT = tc.tensor(dT, device= device0)
            if False:
                def EdXsqZ_Tensor(theta):
                    EdXsqZ = 0
                    step_theta = [None] * n_theta_steps
                    step_theta = tc.full([n_theta_steps , n_strata, n_ass], 0.0, device= device0)
                    for j in range(n_theta_steps):
                        l = int((M ** (j + 0) - 1) / (M - 1))
                        r = int((M ** (j + 1) - 1) / (M - 1) - 1)
                        theta_i0 = theta[l:r +1]
                        step_theta[j] = step_theta[j] + \
                                        theta_i0.repeat_interleave(M ** (n_theta_steps - i), dim=0)
                    strata_theta = torch.transpose(step_theta, 0, 1)
                    EdXsqZ = 0
                    for i in range(n_strata):
                        theta_i = strata_theta[i]
                        integ_theta_dW = tc.sum(theta_i[:, None, :] * dW_stratas[i], dim=0)
                        integ_thetasq_dt = tc.sum((theta_i ** 2) * dT)
                        exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                        EdXsqZ = EdXsqZ + p_strata * tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
                    return EdXsqZ

                def EdXsqZ_TensorGrad(theta):
                    integ_theta_dW = tc.sum(theta[:, None, :] * self.dW, dim=0)
                    integ_thetasq_dt = tc.sum((theta ** 2) * dt)
                    exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                    EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
                    EdXsqZ.backward()
                    return EdXsqZ.data, theta.grad.data
                if False:
                    def EdXsqZ_Tensor(theta):
                        integ_theta_dW = tc.sum(theta[:, None, :] * self.dW, dim=0)
                        integ_thetasq_dt = tc.sum((theta ** 2) * dt)
                        exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                        EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
                        return EdXsqZ

                    def EdXsqZ_TensorGrad(theta):
                        integ_theta_dW = tc.sum(theta[:, None, :] * self.dW, dim=0)
                        integ_thetasq_dt = tc.sum((theta ** 2) * dt)
                        exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                        EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
                        EdXsqZ.backward()
                        return EdXsqZ.data, theta.grad.data

            theta = theta_init
            val, grad = EdXsqZ_TensorGrad(theta_init)
            grad = grad / tc.norm(grad)
            theta = theta_init
            theta.requires_grad = False

            lr = 1
            max_steps = 10
            max_overshots = 30
            steps = 0
            # while (not converged) and (steps < max_steps):
            overshot_counter = 0
            while (steps < max_steps) and (overshot_counter < max_overshots):
                # print(teta)
                proposed_theta = theta - lr * grad
                # print(proposed_teta)
                #proposed_val = EdXsqZ_Tensor(proposed_theta)
                proposed_val = EdXsqZ_TensorGrad(proposed_theta)[0]

                # print(proposed_val)
                if proposed_val < 0.95 * val:
                    theta = proposed_theta
                    # self.teta_path.append(teta)
                    val = proposed_val
                elif overshot_counter < max_overshots:
                    lr = lr / 2
                    overshot_counter += 1
                    theta.requires_grad = True
                    val, grad = EdXsqZ_TensorGrad(theta)
                    grad = grad / tc.norm(grad)
                    theta.requires_grad = False

                steps += 1

            end = time.time()
            # print("GRADDDIENTENVERFAHREN", end - start, "s")
            # print(theta)
            return theta, val





#MLMC = MultiLevelMonteCarlo(option1, asset_coll)
#print(MLMC.sim_mlmc())
#quit()

def multilevel_mc(assets, option,
                n_sims: int, n_batches: int, n_levels: int, n_steps_init: int):  # \

    Integrator1 = Integrator(asset_coll)
    #option = option1
    #       -> tuple[float, float]:
    fp_type = tc.float64
    M = 2
    level_res = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")
    level_var = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")

    theta = tc.tensor([[0]], device = "cuda:0" )
    #theta = None

    teta = 0 #1.25


    n_steps_list = [n_steps_init * (M ** l) for l in range(n_levels)]
    n_batches_list = [math.ceil(n_batches / (M ** l)) for l in range(n_levels)]
    n_batches_list[0] = n_batches_list[0] * 1

    start = time.time()
    for l in range(1  * n_levels):
        S_paths = paths(n_sims, asset_coll, l)

        n_batches = int(n_batches_list[l])
        print(n_sims * n_batches)

        n_steps = int(n_steps_list[l])
        batch_res = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")
        batch_var = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")
        #for j in range(n_batches):
        for j in range(n_batches):
            #tc.cuda.synchronize(device="cuda:0")
            #S_paths = paths(n_sims, asset_coll, l)
            S_paths.reset_tensors(l, option)

            S_paths = Integrator1.sim_paths_layered(S_paths, option, n_steps * 1, option.expiry, theta =theta)
            #tc.cuda.synchronize(device="cuda:0")

            print(S_paths.optimize_theta(option = option))
            #quit()

            batch_var[j], batch_res[j] = S_paths.var_mean(option = option)
            #var, res = S_paths.var_mean(option=option)
            #level_var[l] += var
            #level_res[l] += res
            #print(batch_res[j])
            tc.cuda.synchronize(device="cuda:0")
        level_var[l], level_res[l] = tc.mean(batch_var), tc.mean(batch_res)
        print(level_res[l], level_var[l])
        #level_var[l], level_res[l] = level_var[l] / n_batches, level_res[l] / n_batches
    end = time.time()
    print(end - start,"s", "asdasdasdad")
    #print
    #quit()

    Res = float(tc.sum(level_res, dim=0))
    #Res = float(tc.sum(level_res[:1], dim=0))
    Var = float(tc.sum(level_var, dim=0))
    r = assets.risk_free
    T = option.expiry
    return math.exp(- r * T) * Res, Var

#class StandardMonteCarlo():




if False:
    start = time.time()
    tc.cuda.synchronize(device = "cuda:0")
    #E, V = multilevel_mc(90, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** 0)), int(2 ** 12), 8, 1 )
    E, V = multilevel_mc(asset_coll, option1, int((2 ** 20) * (2 ** 0)), int(2 ** 12), 8, 1 )
    print(E, V)
    end = time.time()
    print(end - start, "s")


    #E, V = multilevel_mc(100, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** 0)), int(2 ** 0), 1, 2 ** 14)
    E, V = multilevel_mc(100, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** 0)), int(2 ** 14), 16, 1 )
    print(colored((abs(3.294086516281595 - E), V ), "green"))
    print(E)
    #E, V = multilevel_mc(90, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** 0)), int(2 ** 0), 1, 2 ** 14)
    E, V = multilevel_mc(90, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** 0)), int(2 ** 14), 16, 1 )
    print(colored((abs(1.822512255945242 - E), V), "green"))
    print(E)
    #E, V = multilevel_mc(90, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** -4)), int(2 ** 16), 16, 1 )
    #E, V = multilevel_mc(90, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** 0)), int(2 ** 12), 18, 1 )
    #print(abs(3.294086516281595 - E), V )
    #print(1.822512255945242 - E, V)
    # print(2.758443856146076 - E, V)

    # 3.221591131246868
    print(E)
    tc.cuda.synchronize(device = "cuda:0")
    end = time.time()
    print(end - start, "s")

    quit()

    start = time.time()
    print(2.758443856146076 - crude_mc(90, 0.03, 0.15, 1.0, 100, int((2 ** 20) * (2 ** 3)), int(2 ** 4), 1 ))
    end = time.time()
    print(end - start, "s")

    start = time.time()
    print(2.758443856146076 - crude_mc(90, 0.03, 0.15, 1.0, 100, int(10 ** 8), 10, 128 ))
    end = time.time()
    print(end - start, "s")

    start = time.time()
    print(2.758443856146076 - crude_mc(90, 0.03, 0.15, 1.0, 100, int(10 ** 6), 100, 128 ))
    end = time.time()
    print(end - start, "s")

    start = time.time()
    print(2.758443856146076 - crude_mc(90, 0.03, 0.15, 1.0, 100, int(10 ** 8), 1, 128 ))
    end = time.time()
    print(end - start, "s")

    start = time.time()
    print(2.758443856146076 - crude_mc(90, 0.03, 0.15, 1.0, 100, int(10 ** 8), 10, 1 ))
    end = time.time()
    print(end - start, "s")

    start = time.time()
    print( 2.758443856146076 - crude_mc(90, 0.03, 0.15, 1.0, 100, int(10 ** 8), 10, 128 ))
    end = time.time()
    print(end - start, "s")


    print(crude_mc.code)


#@tc.jit.script
def multilevel_mc234(S0: float, mu: float, sigma: float,
             T: float, K: float,
             n_sims: int, n_batches: int, n_levels: int, n_steps_init: int): # \
             #       -> tuple[float, float]:
    fp_type = tc.float64
    M = 2
    level_res = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")
    level_var = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")
    level_res1 = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")
    level_var1 = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")
    level_res2 = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")
    level_var2 = tc.full([n_levels], 0.0, dtype=fp_type, device="cuda:0")
    teta = 1.25

    S_t = tc.empty([n_sims], dtype=fp_type, device="cuda:0")
    S_t_sub = tc.empty([n_sims], dtype=fp_type, device="cuda:0")
    dW = tc.empty([n_sims], dtype=fp_type, device="cuda:0")
    dW_sub = tc.empty([n_sims], dtype=fp_type, device="cuda:0")
    empty_tensor = tc.empty([n_sims], dtype=fp_type, device="cuda:0")
    W_t = tc.empty([n_sims], dtype=fp_type, device="cuda:0")


    B = 125
    b = np.sign(S0 - B)
    print(b)
    #b = 1
    S_was_barrier_hit = tc.empty([n_sims], device="cuda:0", dtype=tc.bool)
    S_sub_was_barrier_hit = tc.empty([n_sims], device="cuda:0", dtype=tc.bool)


    n_steps_list = [n_steps_init * (M ** l) for l in range(n_levels)]
    n_batches_list = [math.ceil(n_batches / (M ** l) ) for l in range(n_levels)]
    n_batches_list[0] = n_batches_list[0]  * 1

    for l in range(n_levels):
        tc.cuda.synchronize(device="cuda:0")
        start = timing()
        n_steps = int(n_steps_list[l])
        n_batches = int(n_batches_list[l])
        is_coarse_step_list = [((step_i % M == M - 1) and step_i > 0) for step_i in range(n_steps)]

        dt, dt_sub = T/n_steps, ((T/n_steps) * M)

        sqrt_dt, sqrt_dt_sub = math.sqrt(dt), math.sqrt(dt_sub)
        batch_res = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")
        batch_var = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")
        batch_res1 = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")
        batch_var1 = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")
        batch_res2 = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")
        batch_var2 = tc.full([n_batches], 0.0, dtype=fp_type, device="cuda:0")

        for j in range(n_batches):
            tc.cuda.synchronize(device = "cuda:0")
            start2 = timing()

            S_t.fill_(S0)
            S_t_sub.fill_(S0)
            dW_sub.fill_(0.0)
            W_t.fill_(0.0)

            S_extra, S_sub_extra = get_extras(S_t)

            S_was_barrier_hit = tc.gt(S_t, S_t + 1 )
            S_sub_was_barrier_hit = tc.gt(S_t_sub, S_t_sub + 1)

            pS_was_barrier_hit = tc.full([n_sims], float(1), device="cuda:0", dtype=tc.bool)
            pS_sub_was_barrier_hit = tc.full([n_sims], float(1), device="cuda:0", dtype=tc.bool)
            pS_sub_acc = tc.full([n_sims], float(1), device="cuda:0", dtype=tc.bool)
            dW_t_curr = dW
            for i in range(n_steps):
                dW_t_prev = dW_t_curr
                dW.normal_().mul_(sqrt_dt)
                #dW_t_curr = dW
                W_t += dW
                dW += teta * dt
                dW_t_curr = dW
                dW_sub += dW

                S_t0 = S_t
                dS = S_t0 * (mu * dt + sigma * dW + 0.5 * (sigma ** 2) * (dW ** 2 - dt))
                #dS = S_t0 * (mu * dt + sigma * dW )
                S_t1 = S_t0 + dS
                S_t = S_t1

                #pn = tc.exp( tc.divide(2 * tc.clamp((S_t0 - B), min=0) * tc.clamp((S_t1 - B), min=0) , (sigma ** 2) * dt ))
                pn = tc.exp( tc.divide( -2 * tc.clamp(b * (S_t0 - B), min=0) * tc.clamp( b * (S_t1 - B), min=0) , ((sigma * S_t0) ** 2) * dt ))
                # in giles mlmc code https://people.maths.ox.ac.uk/~gilesm/mlqmc/matlab/mcqmc06/mcqmc06.m
                # the max is of the product
                #pn = tc.exp( tc.divide( -2 * tc.clamp(b * (S_t0 - B) * (S_t1 - B), min=0) , ((sigma * S_t0) ** 2) * dt ))

                pS_was_barrier_hit =  pS_was_barrier_hit * (1 - pn)
                #pS_was_barrier_hit = pS_was_barrier_hit * pn
                pS_sub_acc = pS_sub_acc * pn
                #print(tc.mean(pn), tc.mean(1 - pn))
                #pS_was_barrier_hit = pS_was_barrier_hit * (1 - pn)

                S_was_barrier_hit = tc.gt(1 * (S_t - B), 0) + S_was_barrier_hit
                #print(S_t[0], S_was_barrier_hit[0], B)

                if is_coarse_step_list[i]:
                    S_t0_sub = S_t_sub
                    dS_sub = S_t0_sub * (mu * dt_sub + sigma * dW_sub + 0.5 * (sigma ** 2) * (dW_sub ** 2 - dt_sub))
                    #dS_sub = S_t0_sub * (mu * dt_sub + sigma * dW_sub )
                    S_t1_sub = S_t0_sub + dS_sub
                    S_t_sub = S_t1_sub

                    #S_t05 = 0.5 * (S_t0_sub + S_t1_sub + sigma * S_t0_sub * math.sqrt(dt_sub) * empty_tensor.normal_() )
                    #S_t05 = S_t0_sub + 0.5 * ( S_t1_sub - S_t0_sub )
                    #S_t05 = S_t0_sub + 0.5 * (S_t1_sub - S_t0_sub) # + sigma * S_t0_sub * (dW_t_curr - dW_t_prev )
                    #S_t05 =  0.5 * (  S_t0_sub + S_t1_sub  + sigma * S_t0_sub * ( dW_t_prev - dW_t_curr ) )
                    S_t05 =  0.5 * (  S_t0_sub + S_t1_sub  + sigma * S_t0_sub * ( - dW_t_prev + dW_t_curr ) )
                    #S_t05 = 0.5 * (S_t0_sub + S_t1_sub)

                    #pn_t00_sub = tc.exp(tc.divide(-2 * tc.clamp(b * (S_t0_sub - B), min=0) * tc.clamp( b * (S_t05 - B), min=0),
                    #                      ((sigma * S_t0_sub) ** 2) * dt))
                    #pn_t01_sub = tc.exp(tc.divide(-2 * tc.clamp( b * (S_t05 - B), min=0) * tc.clamp( b * (S_t1_sub - B), min=0),
                    #                      ((sigma * S_t0_sub) ** 2) * dt))
                    #pn_t00_sub = tc.exp(tc.divide(-2 * tc.clamp((S_t0_sub - B) * (S_t05 - B), min=0),
                    #                      ((sigma * S_t0_sub) ** 2) * dt))
                    #pn_t01_sub = tc.exp(tc.divide(-2 * tc.clamp((S_t05 - B) * (S_t1_sub - B), min=0),
                    #                      ((sigma * S_t0_sub) ** 2) * dt))
                    pn_sub = tc.exp(tc.divide(-2 * tc.clamp( b * (S_t0_sub - B), min=0) * tc.clamp( b * (S_t1_sub - B), min=0),
                                          ((sigma * S_t0_sub) ** 2) * dt_sub))

                    #pS_sub_was_barrier_hit = pS_sub_was_barrier_hit * (1 - pn_t00_sub) * (1 - pn_t01_sub)
                    pS_sub_was_barrier_hit = pS_sub_was_barrier_hit * (1 - pn_sub)
                    #pS_sub_was_barrier_hit = pS_sub_was_barrier_hit * (pn_t00_sub * pn_t01_sub)
                    #pS_sub_was_barrier_hit = pS_sub_was_barrier_hit * pS_sub_acc
                    #S_t_sub += S_t_sub * (mu * dt_sub + sigma * dW_sub + 0.5 * (sigma ** 2) * (dW_sub ** 2 - dt_sub))

                    S_sub_was_barrier_hit = tc.gt(1 * (S_t_sub - B), 0)  + S_sub_was_barrier_hit
                    dW_sub.fill_(0.0)
                    pS_sub_acc.fill_(1.0)

            Z = tc.exp(- teta * W_t - 0.5 * (teta ** 2) * T )


            P = payoff_barrier(S_t, S_was_barrier_hit, K, sigma, dt )
            prod_p = pS_was_barrier_hit
            P1 = payoff_barrier1(S_t, prod_p, K) #, sigma, dt )
            P2 = payoff_barrier2(S_t, prod_p, K, sigma, dt)  # , sigma, dt )

            if l > 0:
                P_sub = payoff_barrier(S_t_sub, S_sub_was_barrier_hit, K, sigma, dt_sub)
                Res =  tc.var_mean((P - P_sub) * Z, dim=0)

                prod_p_sub = pS_sub_was_barrier_hit
                P1_sub = payoff_barrier1(S_t_sub, prod_p_sub, K)
                Res1 = tc.var_mean((P1 - P1_sub) * Z, dim=0)
                P2_sub = payoff_barrier2(S_t_sub, prod_p_sub, K, sigma, dt_sub)
                Res2 = tc.var_mean((P2 - P2_sub) * Z, dim=0)
                if j == 0:
                    print(colored("some means for NOT hit", "blue"))
                    print("indicator mean", 1 - tc.count_nonzero(S_was_barrier_hit) / n_sims)
                    print("prod p mean", tc.mean(pS_was_barrier_hit))
                    print("prod p", pS_was_barrier_hit)
                    print("indicator mean", 1 - tc.count_nonzero(S_sub_was_barrier_hit) / n_sims)
                    print("prod p sub mean", tc.mean(pS_sub_was_barrier_hit))
                    print("prod p sub", pS_sub_was_barrier_hit)
                    #print(tc.var_mean((P1_sub) * Z, dim=0), "level", l, "P1_sub var mean")
                    #print(tc.var_mean((P1) * Z, dim=0), "level", l, "P1 var mean")
            else:
                Res = tc.var_mean((P ) * Z, dim=0)
                Res1 = tc.var_mean((P1 ) * Z, dim=0)
                Res2 = tc.var_mean((P2 ) * Z, dim=0)
                if j == 0:
                    print(colored("some means for NOT hit", "blue"))
                    print("indicator mean", 1 - tc.count_nonzero(S_was_barrier_hit) / n_sims)
                    print("prod p mean", tc.mean(pS_was_barrier_hit))
                    print("prod p", pS_was_barrier_hit)

            level_var[l], level_res[l] = level_var[l] + Res[0], level_res[l] + Res[1]
            level_var1[l], level_res1[l] = level_var1[l] + Res1[0], level_res1[l] + Res1[1]
            level_var2[l], level_res2[l] = level_var2[l] + Res2[0], level_res2[l] + Res2[1]
            tc.cuda.synchronize(device="cuda:0")
            end2 = timing()

        level_var[l], level_res[l] = level_var[l]/n_batches, level_res[l]/n_batches
        level_var1[l], level_res1[l] = level_var1[l] / n_batches, level_res1[l] / n_batches
        level_var2[l], level_res2[l] = level_var2[l] / n_batches, level_res2[l] / n_batches
        tc.cuda.synchronize(device="cuda:0")
        end = timing()

        print(float(level_res[l]), float(level_var[l]), "Glasserman approx")
        print(float(level_res1[l]), float(level_var1[l]), "conditional MC")
        print(float(level_res2[l]), float(level_var2[l]), "conditional MC 2")
        print("Level", l, "/", n_levels - 1,"%", n_batches, "x 2 ^", int(math.log2(n_sims)),
              "~= ", int(n_batches / (2** int(math.log2(n_batches)))),  "x 2 ^",
              int(math.log2(n_batches) + int(math.log2(n_sims))), "sims with",
              n_steps, "steps in", float(end - start), "s")
        #batchvariance mean or sum??
        #print(level_res)

    Res = float(tc.sum(level_res, dim=0))
    Var = float(tc.sum(level_var, dim=0))
    Res1 = float(tc.sum(level_res1, dim=0))
    Var1 = float(tc.sum(level_var1, dim=0))
    Res2 = float(tc.sum(level_res2, dim=0))
    Var2 = float(tc.sum(level_var2, dim=0))
    print("conditional MC", math.exp(- mu * T ) * Res1, Var1)
    print("conditional MC 2", math.exp(- mu * T) * Res2, Var2)
    print("glasser approx only", math.exp(- mu * T) * Res, Var)
    return math.exp(- mu * T ) * Res, Var








@tc.jit.ignore()
def timing():
    return tc.tensor([time.time()], dtype= tc.float64)

@tc.jit.script
def payoff( S_t: tc.FloatTensor, K: float):
    return tc.clamp(S_t - K, min=0)

@tc.jit.script
def payoff_barrier( S_t: tc.FloatTensor, S_barrier_was_hit: tc.BoolTensor, K: float, sigma: float, dt: float):
    b = -1
    beta = 0.582597
    cc = math.exp( b * beta * sigma * math.sqrt(dt))
    return tc.clamp((S_t - K * cc), min=0) * tc.logical_not(S_barrier_was_hit)

@tc.jit.script
def payoff_barrier1( S_t: tc.FloatTensor, prod_p: tc.BoolTensor, K: float):
    #p = 1 - prod_p
    p = prod_p
    #print("p", p)
    return tc.clamp((S_t - K), min=0) * p

def payoff_barrier2( S_t: tc.FloatTensor, prod_p: tc.BoolTensor, K: float, sigma: float, dt: float):
    #p = 1 - prod_p
    p = prod_p
    b = -1
    beta = 0.582597
    cc = math.exp(b * beta * sigma * math.sqrt(dt))
    return tc.clamp((S_t - K * cc), min=0) * p


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class UnHiddenPrints:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

#print(multilevel_mc.code)