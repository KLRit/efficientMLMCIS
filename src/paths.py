import torch as tc
import time
import math
from utils import *
device0 = "cuda:0"
fp_type =tc.float32


s2_global = 1.0 #1.0
class paths():
    """
    Container class for relevant information about the integrated trajectories.
    To use for evaluation and optimization.
    """
    def __init__(self, n_sims, assets, level, calling_mcalgo = None):
        #self.antithetic = False
        self.M = calling_mcalgo.M
        self.ImportanceSampling = calling_mcalgo.ImportanceSampling
        self.ImportanceSampling_training = self.ImportanceSampling
        self.antithetic = calling_mcalgo.AntitheticSampling
        self.calling_mcalgo = calling_mcalgo
        self.device = device0
        self.level = level
        #self.option = option
        self.assets = assets
        self.n_assets = self.assets.n_assets
        self.n_sims = n_sims

        self.n_sims_sub = self.n_sims
        self.n_slice = self.n_sims
        self.n_av = int(self.M)
        if self.antithetic:
            self.n_sims_sub = int(self.n_sims / self.n_av)
            if self.level > 0:
                self.n_slice = self.n_sims_sub
        # self.dim = [self.n_sims, self.n_assets]
        # self.dim_sub = [self.n_sims, self.n_assets]


        self.S_t = tc.empty(self.dim, device=device0)
        self.dW_t = tc.empty(self.dim, device=device0)
        #self.S_t = tc.tensor(self.assets.S0, device=device0).repeat(n_sims, 1)
        #print(self.S_t.shape)
        #if level > 0:
        self.S_t_sub = tc.empty(self.dim_sub, device=device0)
        #else:
            # dummy Tensor for simpler compatibility with eg the reset method
        #    self.S_t_sub = tc.empty([1, self.n_assets], device=device0)
            #self.S_t_sub = tc.tensor(self.assets.S0, device=device0).repeat(n_sims, 1)
        self.integ_theta_dW = tc.tensor([0], device=device0)
        #self.integ_theta_dW = tc.tensor(self.dim, device=device0)
        self.integ_thetasq_dt = tc.tensor([0], device=device0)
        self.t = 0
        self.n_steps = 0
        self.step_i = 0
        self.t_sub = 0
        self.n_steps_sub = 0
        self.step_i_sub = 0
        #self.n_steps = n_steps

        self.t = 0
        #self.S_t = tc.tensor(option.S0, device=device0, dtype=fp_type).repeat(n_sims, 1).transpose(-1, 0)
        #self.S_extra = []


        #self.S_t_sub = tc.tensor(option.S0, device=device0, dtype=fp_type).repeat(n_sims, 1).transpose(-1, 0)
        #self.S_extra_sub = []

        self.is_importance_sampled = False

        self.stratum_i = None

    @property
    def dim(self):
        return [self.n_sims, self.n_assets]

    @property
    def dim_sub(self):
        return [self.n_sims_sub, self.n_assets]

    def reset_tensors(self, level, mc_calling):
        """Reset (or initialize) tensor values for integration."""
        self.level = level
        l = level
        self.option = mc_calling.option
        self.M = mc_calling.M
        self.n_steps = mc_calling.n_steps[l]
        #print(l, self.n_steps)
        self.n_steps_sub = mc_calling.n_steps_sub[l]
        self.dt = mc_calling.dt[l]
        self.sqrt_dt = mc_calling.sqrt_dt[l]
        if level > 0:
            self.dt_sub = mc_calling.dt_sub[l]
            self.sqrt_dt_sub = mc_calling.sqrt_dt_sub[l]

        self.t = 0
        self.step_i = 0
        self.t_sub = 0
        self.step_i_sub = 0

        self.dW_T = 0
        self.dW_theta_T = 0
        self.T = mc_calling.T
        self.theta = mc_calling.Theta[level]
        #self.theta = mc_calling.Theta[0]

        if level == 0 or not self.antithetic:
            self.n_slice = self.n_sims
        elif level > 0 and self.antithetic:
            self.n_slice = self.n_sims_sub

        for i in range(len(self.assets.S0)):
            self.S_t[:,i].fill_(self.assets.S0[i])
            if level > 0:
                self.S_t_sub[:,i].fill_(self.assets.S0[i])
        self.integ_theta_dW.fill_(0)
        self.integ_thetasq_dt.fill_(0)
        if self.option.is_path_dependent:
            #print("resetting paths")
            self.option.reset_paths(self)

        #self.theta123 = 0.0



    def dX(self, option = None):
        """Return difference of payoffs between fine and (if exists) coarse level."""
        if option == None:
            option =self.option

        if self.level > 0:
            #print(self.S_t, self.S_t_sub)
            if not self.antithetic:
                dX = (option.payoff(self) - option.payoff(self, sub = True))
                #print(dX.shape, "dX shape")
            elif self.antithetic:
                s = slice(0, self.n_slice)
                dX = (tc.mean((option.payoff(self)).view([self.n_av] + [self.n_sims_sub, 1]),0 ) - option.payoff(self, sub=True, slicing = s))

        elif self.level == 0:
            dX = option.payoff(self)


        if self.is_importance_sampled:
            dX = dX * self.Z_inv()

        return dX

    #@property
    def var_mean(self, option = None):
        """Return variance and mean tuple of the option payoff evaluation of (multilevel) paths."""
        return tc.var_mean(self.dX(option = option))

    def var_mean_single(self, option = None, sub = False):
        if option == None:
            option =self.option
        X = option.payoff(self, sub = sub)
        if self.is_importance_sampled:
            X = X * (1 / self.Z())
        return tc.var_mean(X)

    def Z234(self):
        """Return the value of the Radon-Nikodym-Derivative Z
        respective to the importance-sampled Wiener paths."""
        return tc.prod(tc.exp( self.integ_theta_dW + 0.5 * self.integ_thetasq_dt), dim = 1, keepdim= True)

    def Z123(self):
        """Return the value of the Radon-Nikodym-Derivative Z
        respective to the importance-sampled Wiener paths."""
        # print(self.n_slice, "n_slice")
        # print(tc.var_mean(self.integ_theta_dW[:self.n_slice]) )
        return tc.prod(tc.exp( self.integ_theta_dW[:self.n_slice]
                               + 0.5 * self.integ_thetasq_dt[:self.n_slice]),
                       dim = 1, keepdim= True)

    def Z(self):
        """Return the value of the Radon-Nikodym-Derivative Z
        respective to the importance-sampled Wiener paths."""
        # print(self.n_slice, "n_slice")
        # print(tc.var_mean(self.integ_theta_dW[:self.n_slice]) )
        integ_theta_dW = self.dW_T[:self.n_slice] * self.theta
        integ_theta_sqdT = (self.theta ** 2 ) * self.T
        return tc.prod(tc.exp( integ_theta_dW
                               + 0.5 * integ_theta_sqdT),
                       dim = 1, keepdim= True)
    def Z_inv(self):
        """Return the inverse value of the Radon-Nikodym-Derivative Z
        respective to the importance-sampled Wiener paths."""
        integ_theta_dW = self.dW_T[:self.n_slice] * self.theta
        integ_theta_sqdT = (self.theta ** 2 ) * self.T
        return tc.prod(tc.exp( -(integ_theta_dW
                               + 0.5 * integ_theta_sqdT)),
                       dim = 1, keepdim= True)

    def Z_inv213(self):
        """Return the inverse value of the Radon-Nikodym-Derivative Z
        respective to the importance-sampled Wiener paths."""
        return tc.prod(tc.exp( - (self.integ_theta_dW[:self.n_slice]
                                  + 0.5 * self.integ_thetasq_dt[:self.n_slice])),
                       dim = 1, keepdim= True)

    def Z_inv_alt(self):
        return tc.prod(tc.exp( - (self.integ_theta_dW[:self.n_slice]
                                  - 0.5 * self.integ_thetasq_dt[:self.n_slice])),
                       dim = 1, keepdim= True)
    def Z_alt(self):
        return tc.prod(tc.exp( self.integ_theta_dW[:self.n_slice]
                               - 0.5 * self.integ_thetasq_dt[:self.n_slice]),
                       dim = 1, keepdim= True)

    def accumulate_rnd_locconst(self, i, dW_ti, theta_ti, dt):
        self.integ_theta_dW = self.integ_theta_dW + dW_ti * theta_ti
        self.integ_thetasq_dt = (theta_ti ** 2) * dt

    def accumulate_rnd_globconst(self, i, dW_ti, theta_ti, dt):
        pass

    def rnd_normal_dW(self, step_i, dt):
        #dW_t = math.sqrt(dt) * tc.normal(0.0, 1.0, self.dim, device=device0)
        #if step_i % self.M == 0 or self.level == 0 or not self.antithetic:
        T = self.calling_mcalgo.option.expiry
        sqrt_T = math.sqrt(T)


        if step_i % self.n_av == 0 or self.level == 0 or not self.antithetic:
            #self.dW_t.normal_().mul_(self.sqrt_dt)
            #self.dW_t = self.bb_gen.__next__().mul_(1)
            use_bb = False
            if not use_bb:
                self.dW_t.normal_().mul_(self.sqrt_dt)
            elif self.antithetic and self.level >0:
                self.dW_t = tc.cat([self.bb_gen.__next__().mul_(sqrt_T) for j in range(min(self.n_steps, self.n_av))], dim = 0)
            else:
                self.dW_t = self.bb_gen.__next__().mul_(sqrt_T)


            dW_t = self.dW_t

            if self.antithetic:
                self.dW_t_store = dW_t.clone()
        elif self.antithetic:
            self.dW_t_store = self.dW_t_store.roll(self.n_sims_sub, 0)
            dW_t = self.dW_t_store


        self.dW_T += dW_t
        if self.ImportanceSampling:

            #theta = self.theta
            dW_t = dW_t + self.theta * dt
            self.is_importance_sampled = True
            #theta = self.calling_mcalgo.Theta[self.level]

        return dW_t

    def optimize_theta(self, option=None, theta_init=None):
        start = time.time()
        dX = self.dX(option=option)

        dXsq = dX ** 2
        T = option.expiry
        n_steps = self.n_steps
        dt = T / n_steps

        constant_theta = self.antithetic
        # constant_theta = False
        constant_theta = True



        if theta_init == None:
            if constant_theta:
                theta_init = tc.tensor([0.0] * self.n_assets, device=device0, requires_grad=True)
            else:
                pass
                #theta_init = tc.full([n_steps, self.n_assets], 0.0, device=device0, requires_grad=True)
            # theta_init = tc.full([min(n_theta_steps, n_steps), self.n_assets], 0.0, device = device0, requires_grad= True)
            # theta_init.fill_(0.0)

        # n_theta_steps = min(n_theta_steps, n_steps)
        # theta_step_sizes = [2, 1, 1]
        # n_strata = [1, 2, 4]
        # theta_dt = theta_step_sizes * dt
        if constant_theta:
            #W_T = tc.sum(self.dW, 0)
            #W_T = self.dW_T[:self.n_slice]
            W_T = self.dW_t[:self.n_slice]

            def EdXsqZ_Tensor(theta):
                integ_theta_dW = theta * W_T
                integ_thetasq_dt = (theta ** 2) * T
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                # print(exp_integs.shape, "shapezzz")
                EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
                #EdXsqZ = tc.mean(dXsq * tc.prod(tc.exp(- theta * W_T + 0.5 * (theta ** 2) * T), dim=1, keepdim=True), dim=0)
                return EdXsqZ

            def EdXsqZ_TensorGrad(theta):
                integ_theta_dW = theta * W_T
                integ_thetasq_dt = (theta ** 2) * T
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                # print(exp_integs.shape)
                # print(tc.prod(exp_integs, dim=1, keepdim=True).shape)
                # print(dXsq.shape)
                EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)

                # print(EdXsqZ, "EdXsqZ")
                # print(theta, "theta")
                EdXsqZ.backward()
                return EdXsqZ.data, theta.grad.data

        def grad_desc(func, arg_init, max_steps=100, lr=0.2):
            i = 0

            factor = 100

            val0 = factor * func(arg_init)
            arg = arg_init
            arg.requires_grad = True



            def func1(arg):
                return factor * func(arg)
            old_val = val0
            while i < max_steps:
                # print("grad desc", i, "///////////////////////////")
                # val, grad = func(arg)
                val = factor * func1(arg)
                print(val)
                val.backward()
                grad = arg.grad
                grad = grad / tc.norm(grad)
                with tc.no_grad():
                    arg = arg - lr * grad
                    lr = 0.5 * lr
                #arg = arg - lr * grad
                arg.requires_grad = True
                i = i + 1

            arg.requires_grad = False
            val1 = func(arg)


            # print(arg_init, "arginit")
            # print(val0, "old val")
            # print(val1, "new val")
            # print(arg, "new arg")

            return arg

        def newton123(func, theta_init, n_steps=100):
            theta_init.requires_grad = True
            for i in range(n_steps):
                previous = theta_init.clone()
                val = func(theta_init)
                val.backward()

                theta_init.data = theta_init.data - (val / theta_init.grad).data

                theta_init.grad.data.zero_()

                if tc.abs(theta_init - previous) < 0.1:
                    return theta_init
            return theta_init

        # def newton(func, guess, threshold=1e-7):
        #     guess.requires_grad = True
        #     value = func(guess)
        #     while abs(value.data.cpu().numpy()[0]) > threshold:
        #         value = func(guess)
        #         value.backward()
        #         guess.data -= (value / guess.grad).data
        #         guess.grad.data.zero_()
        #     return guess.data

        # def line_search(theta_init, func, n_steps = 10):
        #     x = theta_init
        #     val0 = func(x)
        #     val1 = func(x + h)
        #     val2 = func(x - h)
        #     if val1 < val0:
        #         while val1 < val0:
        #             x = x + h
        #             val0 = func(x)
        #


        #theta_init = tc.tensor([0.0] * self.n_assets, device=device0, requires_grad=True)
        theta_init1 = theta_init.clone()
        theta_init1.requires_grad = True
        theta = grad_desc(EdXsqZ_Tensor, theta_init1, max_steps=20, lr=1)
        #theta = newton(EdXsqZ_Tensor,theta_init1)
        theta.requires_grad = False
        #plotshow_func(EdXsqZ_Tensor, mark = theta)
        val = EdXsqZ_Tensor(theta)
        return theta, val


    def optimize_theta123(self, option = None, theta_init = None):
        start = time.time()
        dX = self.dX(option = option)

        dXsq = dX ** 2
        T = option.expiry
        n_steps = self.dW.shape[0]
        dt = T/n_steps

        constant_theta = self.antithetic
        #constant_theta = False
        constant_theta = True
        if theta_init == None:
            if constant_theta:
                theta_init = tc.tensor([0.0] * self.n_assets, device = device0, requires_grad= True)
            else:
                theta_init = tc.full([n_steps, self.n_assets], 0.0, device = device0, requires_grad= True)
            #theta_init = tc.full([min(n_theta_steps, n_steps), self.n_assets], 0.0, device = device0, requires_grad= True)
            #theta_init.fill_(0.0)

        # n_theta_steps = min(n_theta_steps, n_steps)
        # theta_step_sizes = [2, 1, 1]
        # n_strata = [1, 2, 4]
        # theta_dt = theta_step_sizes * dt
        if constant_theta:
            #W_T = self.dW_T[:self.n_slice]
            #W_T = tc.sum(self.dW, 0)
            W_T = self.dW_t[:self.n_slice]
            def EdXsqZ_Tensor(theta):
                integ_theta_dW = theta * W_T
                integ_thetasq_dt = (theta ** 2) * T
                exp_integs = tc.exp( - integ_theta_dW + 0.5 * integ_thetasq_dt )
                #print(exp_integs.shape, "shapezzz")
                EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim =1, keepdim= True), dim = 0)
                return EdXsqZ

            def EdXsqZ_TensorGrad(theta):
                integ_theta_dW = theta * W_T
                integ_thetasq_dt = (theta ** 2) * T
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                #print(exp_integs.shape)
                #print(tc.prod(exp_integs, dim=1, keepdim=True).shape)
                #print(dXsq.shape)
                EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)

                #print(EdXsqZ, "EdXsqZ")
                #print(theta, "theta")
                EdXsqZ.backward()
                return EdXsqZ.data, theta.grad.data

        else:
            def EdXsqZ_Tensor(theta):
                integ_theta_dW = tc.sum( theta[:,None,:] * self.dW, dim = 0)
                integ_thetasq_dt = tc.sum((theta ** 2) * dt)
                exp_integs = tc.exp( - integ_theta_dW + 0.5 * integ_thetasq_dt )
                EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim =1, keepdim= True), dim = 0)
                return EdXsqZ

            def EdXsqZ_TensorGrad(theta):
                integ_theta_dW = tc.sum( theta[:,None,:] * self.dW, dim = 0)
                integ_thetasq_dt = tc.sum((theta ** 2) * dt)
                exp_integs = tc.exp( - integ_theta_dW + 0.5 * integ_thetasq_dt )
                EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim =1, keepdim= True),dim = 0)
                EdXsqZ.backward()
                return EdXsqZ.data, theta.grad.data



        #theta, val = grad_line_search(EdXsqZ_Tensor, theta_init, max_steps=10, max_dirchanges = 10, lr=1)
        #return theta, val

        theta = theta_init
        val, grad = EdXsqZ_TensorGrad(theta_init)

        grad = grad / tc.norm(grad)
        theta = theta_init
        theta.requires_grad = False

        lr = 1
        max_steps = 100 # 10
        max_overshots = 30
        steps = 0
        # while (not converged) and (steps < max_steps):
        overshot_counter = 0
        while (steps < max_steps) and (overshot_counter < max_overshots):
            # print(teta)
            proposed_theta = theta - lr * grad
            # print(proposed_teta)
            proposed_val = EdXsqZ_Tensor(proposed_theta)

            # print(proposed_val)
            if proposed_val < 0.95 * val:
                theta = proposed_theta
                #self.teta_path.append(teta)
                val = proposed_val
            elif overshot_counter < max_overshots:
                # lower learning rate. change direction
                lr = lr / 2
                overshot_counter += 1
                theta.requires_grad = True
                #val, grad = EdXsqZ_TensorGrad(theta)
                val = EdXsqZ_Tensor(theta)
                val.backward()
                grad = theta.grad
                grad = grad / tc.norm(grad)
                theta.requires_grad = False

            steps += 1

        end = time.time()
        #print("GRADDDIENTENVERFAHREN", end - start, "s")
        #print(theta)
        return theta, val

    def init_tensors(self, level, n_sims, assets, option):
        """deprecate"""
        self.n_sims = n_sims
        self.S_t = tc.empty([n_sims, len(assets.S0)])
        if level > 0:
            self.S_t_sub = tc.empty([n_sims, len(assets.S0)])
        self.integ_theta_dW = tc.tensor([0])
        self.integ_thetasq = tc.tensor([0])
        self.t = 0
        self.t_sub = 0
    #@property
    #def integ_theta_dW(self):
    #    pass


class paths_Heston(paths):
    def __init__(self, n_sims, assets, level, calling_mcalgo=None):
        super().__init__(n_sims, assets, level, calling_mcalgo = calling_mcalgo)

        self.v_t = tc.empty([n_sims, self.n_assets], device=device0)
        #self.v_t_sub = tc.empty([n_sims, self.n_assets], device=device0)
        self.v_t_sub = tc.empty([self.n_sims_sub, self.n_assets], device=device0)

        self.integ_voltheta_dWv = tc.tensor([0], device=device0)
        self.integ_volthetasq_dt = tc.tensor([0], device=device0)

    def rnd_normal_dWs_dWv(self, step_i, dt):
        #dW_t = math.sqrt(dt) * tc.normal(0.0, 1.0, self.dim, device=device0)
        #if step_i % self.M == 0 or self.level == 0 or not self.antithetic:
        #rho = self.calling_mcalgo.asset_coll.rho
        #sqrt_one_minus_rhosq = self.calling_mcalgo.asset_coll.sqrt_one_minus_rhosq
        use_bb = False
        if use_bb and step_i == 0:
            if  self.level > 0 and self.antithetic:
                self.bb_gen = yield_brownian_bridge_increments(self.dim_sub, self.n_steps)
            else:
                self.bb_gen = yield_brownian_bridge_increments(self.dim, self.n_steps)

        if step_i % self.n_av == 0 or self.level == 0 or not self.antithetic:
            #self.dW_t.normal_().mul_(self.sqrt_dt)
            self.dW_t.normal_().mul_(self.sqrt_dt)
            #self.dW_t = normalu(self.dim).mul_(self.sqrt_dt)
            dW_t = self.dW_t
            #self.dWv_t = rho * dW_t \
            #        + sqrt_one_minus_rhosq * normalu(self.dim).mul_(self.sqrt_dt) # tc.normal(0.0, 1.0, S_paths.dim, device=device0).mul_(sqrt_dt)
            #dWv_t = self.dWv_t
            #dW_t = self.sqrt_dt * tc.normal(0.0, 1.0, self.dim, device=device0)
            #dW_t = math.sqrt(dt) * tc.normal(0.0, 1.0, self.dim, device=device0)
            if self.antithetic:
                self.dW_t_store = dW_t.clone()
                #self.dWv_t_store = dWv_t.clone()
        elif self.antithetic:
            self.dW_t_store = self.dW_t_store.roll(self.n_sims_sub, 0)
            dW_t = self.dW_t_store
            #self.dWv_t_store = self.dWv_t_store.roll(self.n_sims_sub, 0)
            #dWv_t = self.dWv_t_store

        self.dW_T = self.dW_T + dW_t
        #self.ImportanceSampling = self.is_importance_sampled
        # self.ImportanceSampling = False
        # self.is_importance_sampled = False
        if self.ImportanceSampling:

            theta = self.theta

            dW_t = dW_t + theta * dt
            self.is_importance_sampled = True


        rho = self.calling_mcalgo.asset_coll.rho
        sqrt_one_minus_rhosq = self.calling_mcalgo.asset_coll.sqrt_one_minus_rhosq
        if step_i % self.n_av == 0 or self.level == 0 or not self.antithetic:
            testvol = False
            if testvol and self.is_importance_sampled:
                voltheta = +0.5
                dWv_t1.normal_().mul_(self.sqrt_dt)
                # print("???ßß", theta)

                self.integ_voltheta_dWv = self.integ_voltheta_dWv + voltheta * dWv_t1
                self.integ_volthetasq_dt = self.integ_volthetasq_dt + (voltheta ** 2) * dt

                dWv_t0 = dWv_t1 + voltheta * dt
            else:
                dWv_t0 =  normalu(self.dim).mul_( self.sqrt_dt)

            self.dWv_t = rho * dW_t \
                         + sqrt_one_minus_rhosq * dWv_t0
                         #+ sqrt_one_minus_rhosq * normalu(self.dim).mul_( self.sqrt_dt)  # tc.normal(0.0, 1.0, S_paths.dim, device=device0).mul_(sqrt_dt)
            dWv_t = self.dWv_t
            if self.antithetic:
                self.dWv_t_store = dWv_t.clone()
        elif self.antithetic:
            self.dWv_t_store = self.dWv_t_store.roll(self.n_sims_sub, 0)
            dWv_t = self.dWv_t_store


        return dW_t, dWv_t

    def Z123(self):
        """Return the value of the Radon-Nikodym-Derivative Z and Zv
        respective to the importance-sampled Wiener paths."""
        return tc.prod(tc.exp( self.integ_theta_dW[:self.n_slice] + 0.5 * self.integ_thetasq_dt #[:self.n_slice]
                               + self.integ_voltheta_dWv[:self.n_slice] + 0.5 * self.integ_volthetasq_dt), dim = 1, keepdim= True)

    def Z123(self):
        """Return the value of the Radon-Nikodym-Derivative Z and Zv
        respective to the importance-sampled Wiener paths."""
        integ_theta_dW = self.dW_T[:n_slice]
        integ_thetasq_dt = self.theta * self.T

        integ_volthetasq_dt = self.voltheta * self.T
        integ_voltheta_dW = self.dWv_T[:n_slice]
        return tc.prod(tc.exp( self.integ_theta_dW[:self.n_slice] + 0.5 * self.integ_thetasq_dt #[:self.n_slice]
                               + self.integ_voltheta_dWv[:self.n_slice] + 0.5 * self.integ_volthetasq_dt), dim = 1, keepdim= True)


    def reset_tensors(self, level, mc_calling):
        """Reset (or initialize) tensor values for integration."""
        self.level = level
        l = level
        self.option = mc_calling.option
        self.M = mc_calling.M
        self.n_steps = mc_calling.n_steps[l]
        # print(l, self.n_steps)
        self.n_steps_sub = mc_calling.n_steps_sub[l]
        self.dt = mc_calling.dt[l]
        self.sqrt_dt = mc_calling.sqrt_dt[l]
        if level > 0:
            self.dt_sub = mc_calling.dt_sub[l]
            self.sqrt_dt_sub = mc_calling.sqrt_dt_sub[l]

        self.t = 0
        self.step_i = 0
        self.t_sub = 0
        self.step_i_sub = 0

        self.dW_T = 0
        self.T = self.calling_mcalgo.T
        self.theta = self.calling_mcalgo.Theta[level]

        if level == 0 or not self.antithetic:
            self.n_slice = self.n_sims
        elif level > 0 and self.antithetic:
            self.n_slice = self.n_sims_sub

        for i in range(len(self.assets.S0)):
            self.S_t[:,i].fill_(self.assets.S0[i])
            self.v_t[:, i].fill_(self.assets.V0[i])
            if level > 0:
                self.S_t_sub[:, i].fill_(self.assets.S0[i])
                self.v_t_sub[:,i].fill_(self.assets.V0[i])
        self.integ_theta_dW.fill_(0.0)
        self.integ_thetasq_dt.fill_(0.0)
        self.integ_voltheta_dWv.fill_(0.0)
        self.integ_volthetasq_dt.fill_(0.0)

        if self.option.is_path_dependent:
            self.option.reset_paths(self)



    def Z_alt(self):
        """Return the value of the Radon-Nikodym-Derivative Z and Zv
        respective to the importance-sampled Wiener paths."""
        return tc.prod(tc.exp( self.integ_theta_dW + 0.5 * self.integ_thetasq_dt
                               + self.integ_voltheta_dWv + 0.5 * self.integ_volthetasq_dt), dim = 1, keepdim= True)

    def Z_alt_inv(self):
        """Return the inverse value of the Radon-Nikodym-Derivative Z and Zv
        respective to the importance-sampled Wiener paths."""
        return tc.prod(tc.exp( - (self.integ_theta_dW + 0.5 * self.integ_thetasq_dt
                              + self.integ_voltheta_dWv + 0.5 * self.integ_volthetasq_dt)), dim = 1, keepdim=True)

    def optimize_theta1(self, option=None, theta_init=None, voltheta_init = None):
        start = time.time()
        dX = self.dX(option=option)

        dXsq = dX ** 2
        T = option.expiry
        n_steps = self.n_steps
        dt = T / n_steps

        constant_theta = self.antithetic
        # constant_theta = False
        constant_theta = True
        if theta_init == None:
            if constant_theta:
                theta_init = tc.tensor([0.0] * self.n_assets, device=device0, requires_grad=True)
            else:
                pass
                #theta_init = tc.full([n_steps, self.n_assets], 0.0, device=device0, requires_grad=True)
            # theta_init = tc.full([min(n_theta_steps, n_steps), self.n_assets], 0.0, device = device0, requires_grad= True)
            # theta_init.fill_(0.0)

        # n_theta_steps = min(n_theta_steps, n_steps)
        # theta_step_sizes = [2, 1, 1]
        # n_strata = [1, 2, 4]
        # theta_dt = theta_step_sizes * dt
        if constant_theta:
            #W_T = tc.sum(self.dW, 0)
            #W_T = self.dW_T[:self.n_slice]
            W_T = self.dW_t[:self.n_slice]

            def EdXsqZ_Tensor(theta):
                integ_theta_dW = theta * W_T
                integ_thetasq_dt = (theta ** 2) * T
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                # print(exp_integs.shape, "shapezzz")
                #EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)
                EdXsqZ = tc.mean(dXsq * tc.prod(tc.exp(- theta * W_T + 0.5 * (theta ** 2) * T), dim=1, keepdim=True), dim=0)
                return EdXsqZ

            def EdXsqZ_TensorGrad(theta):
                integ_theta_dW = theta * W_T
                integ_thetasq_dt = (theta ** 2) * T
                exp_integs = tc.exp(- integ_theta_dW + 0.5 * integ_thetasq_dt)
                # print(exp_integs.shape)
                # print(tc.prod(exp_integs, dim=1, keepdim=True).shape)
                # print(dXsq.shape)
                EdXsqZ = tc.mean(dXsq * tc.prod(exp_integs, dim=1, keepdim=True), dim=0)

                # print(EdXsqZ, "EdXsqZ")
                # print(theta, "theta")
                EdXsqZ.backward()
                return EdXsqZ.data, theta.grad.data

        def grad_desc(func, arg_init, max_steps=100, lr=0.2):
            i = 0

            factor = 10000

            val0 = factor * func(arg_init)
            arg = arg_init
            arg.requires_grad = True



            def func1(arg):
                return factor * func(arg)
            old_val = val0
            while i < max_steps:
                # print("grad desc", i, "///////////////////////////")
                # val, grad = func(arg)
                val = factor * func(arg)
                print(val)
                val.backward()
                grad = arg.grad
                grad = grad / tc.norm(grad)
                with tc.no_grad():
                    arg = arg - lr * grad
                    lr = 0.75 * lr
                #arg = arg - lr * grad
                arg.requires_grad = True
                i = i + 1

            arg.requires_grad = False
            val1 = func(arg)

            # print(arg_init, "arginit")
            # print(val0, "old val")
            # print(val1, "new val")
            # print(arg, "new arg")

            return arg

        #theta_init = tc.tensor([0.0] * self.n_assets, device=device0, requires_grad=True)
        theta_init1 = theta_init.clone()
        theta_init1.requires_grad = True
        theta = grad_desc(EdXsqZ_Tensor, theta_init1, max_steps=20, lr=1)
        theta.requires_grad = False
        val = EdXsqZ_Tensor(theta)
        return theta, val
#b = 0.25
def normalu(dim, a = float(2 ** -12), b = float(1- (2 ** - 12)), device = "cuda:0"):
    #return tc.normal(0.0, 1.0, dim, device= device0)
    a = max(a, float(2 ** -12))
    b = min(float(1 - (2 ** - 12)), b)
    #U = tc.empty(dim, device=device0)
    #return tc.erfinv(tc.empty_like(X).uniform_().mul_(2).sub_(1)).mul_(math.sqrt(2))
    #return tc.erfinv(tc.empty(dim, device=device, dtype=fp_type).uniform_().mul_(2).sub_(1)).mul_(math.sqrt(2))
    #return tc.erfinv(tc.empty(dim, device=device, dtype=fp_type).uniform_(0.0, 0.5).mul_(2).sub_(1)).mul_(math.sqrt(2))
    #return tc.erfinv(tc.empty(dim, device=device, dtype=fp_type).uniform_(0.0, 1.0).mul_(2).sub_(1)).mul_(math.sqrt(2))
    return tc.erfinv(tc.empty(dim, device=device, dtype=fp_type).uniform_(a , b).mul_(2).sub_(1)).mul_(math.sqrt(2))


def yield_brownian_bridge_increments(dim, n_steps, stratum_i = None):
    strata_partitions4 = [[(0.0, 0.5), (0.0, 0.5) ], [(0.0, 0.5), (0.5, 1.0) ], [(0.5, 1.0), (0.0, 0.5) ], [(0.5, 1.0), (0.5, 1.0) ]]
    strata_partitions4 = [[(0.0, 0.25), (0.0, 1.0) ], [(0.25, 0.5), (0.0, 1.0) ], [(0.5, 0.75), (0.0, 1.0) ], [(0.75, 1.0), (0.0, 1.0) ]]
    strata_partitions2 = [[(0.5, 1.0), (0.0, 1.0) ], [(0.0, 0.5), (0.0, 1.0) ]]
    #strata_partitions2 = [[(0.0, 1.0), (0.0, 1.0)], [(0.0, 1.0), (0.0, 1.0)]]
    strata_partitions1 = [[(0.0, 1.0), (0.0, 1.0)]]
    if stratum_i == None:
        a1, b1 = 0.0, 1.0
        #a1, b1 = 0.5, 1.0
        #a1, b1 = 0.0, 0.5
        a2, b2 = 0.0, 1.0
    else:
        a1, b1 = strata_partitions4[stratum_i][0]
        a2, b2 = strata_partitions4[stratum_i][1]

    #X = tc.empty([n_sims, n_assets], device=device0)
    backward = True
    s2 = s2_global # 1.0
    yield_brownian_bridge_increments.s2 = s2
    if n_steps >= 2:
        if backward:
            #B_T11 = gen_normalu(X, a=a2, b=b2)
            B_T11 = normalu(dim, a = a2, b = b2) * s2
            #B_T01 = 0.5 * B_T11 + math.sqrt(0.5 / 2) * gen_normalu(X, a=a1, b=b1)
            B_T01 = 0.5 * B_T11 + math.sqrt(0.5 / 2) * normalu(dim, a = a1, b = b1)
        else:
            #B_T01 = math.sqrt(0.5) * gen_normalu(X, a=a1, b=b1)
            B_T01 = math.sqrt(0.5) * normalu(dim, a = a1, b = b1)
            #B_T11 = B_T01 + math.sqrt(0.5) *  gen_normalu(X, a=a2, b=b2)
            B_T11 = B_T01 + math.sqrt(0.5) * normalu(dim, a = a2, b = b2)
        yield_brownian_bridge_increments.Z_T = B_T11
        yield B_T11
        if n_steps == 2:
            yield B_T01
            yield B_T11 - B_T01
        else:
            step_i = 0
            t = 0
            Bl = 0
            Br = B_T01
            Bt = 0
            n11 = n_steps
            n01 = int(n_steps/2 )

            dt = 1/n_steps

            while step_i < n01 - 1:
                #mean_factor = ( dt / (  (n01 - step_i) * dt ) )
                mean_factor = 1 / (n01 - step_i)
                var = ( ((n01 - step_i - 1 ) * dt) / (n01 - step_i) )
                #print(mean_factor, var)
                #B_next =  Bl + mean_factor * (Br - Bl) + math.sqrt(std) * tc.normal(0.0, 1.0,[n_sims, n_assets], device=device0)
                #dB =  mean_factor * (Br - Bl) + math.sqrt(var) * tc.normal(0.0, 1.0,[n_sims, n_assets], device=device0)
                dB = mean_factor * (Br - Bl) + math.sqrt(var) * tc.normal(0.0, 1.0, dim, device=device0)
                yield dB
                Bl = Bl + dB
                step_i = step_i + 1
                t = t + dt
            if step_i == n01 - 1:
                dB = Br - Bl
                yield dB
                Br = B_T11
                Bl = B_T01
                step_i = step_i + 1
                t = t + dt
            while step_i < n11 - 1:
                #mean_factor = ( dt / (  (n01 - step_i) * dt ) )
                mean_factor = 1 / (n11 - step_i)
                var = ( ((n11 - step_i - 1) * dt) / (n11 - step_i) )
                #print(mean_factor, var)
                #B_next =  Bl + mean_factor * (Br - Bl) + math.sqrt(std) * tc.normal(0.0, 1.0,[n_sims, n_assets], device=device0)
                #dB =  mean_factor * (Br - Bl) + math.sqrt(var) * tc.normal(0.0, 1.0,[n_sims, n_assets], device=device0)
                dB =  mean_factor * (Br - Bl) + math.sqrt(var) * tc.normal(0.0, 1.0, dim , device=device0)

                yield dB
                Bl = Bl + dB
                step_i = step_i + 1
                t = t + dt
            if step_i == n11 - 1:
                dB = Br - Bl
                yield dB
                Br = B_T11
                Bl = B_T01
    elif n_steps == 1:
        #B_T = gen_normalu(X, a=a1, b=b1)
        B_T = normalu(dim, a = a1, b = b1) * s2
        yield_brownian_bridge_increments.Z_T = B_T
        yield B_T
        yield B_T