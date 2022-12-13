from trash.distributions import *

device0 = "cuda:0"

class Integrator():
    def __init__(self, assets, M = 2, ImportanceSampling = False, discretization = "Euler"):
        self.x123 = 23
        self.M = M
        self.assets = assets
        self.ImportanceSampling = ImportanceSampling

        #self.sampler = antithetic_normal_sampler(device = device0)
        #self.sampler = normal_sampler(device = device0)

        if self.assets.correlated:
            self.correlate_samples = True
            self.corr_cholmat_L = self.assets.corr_cholmat_L
            self.corr_cholmat_R = self.assets.corr_cholmat_R
        else:
            self.correlate_samples = False

        #discretization = "Milstein"
        discretization = "Euler"
        if discretization == "Milstein":
            self.disc_step = assets.Milstein_step
            self.disc_step_sub = assets.Milstein_step
        if discretization == "Euler":
            self.disc_step = assets.Euler_step
            self.disc_step_sub = assets.Euler_step

        self.disc_step1 = assets.Milstein_step
        self.disc_step2 = assets.Euler_step

        self.ImportanceSampling_training = self.ImportanceSampling

    def sim_paths_layered(self, S_paths, option, n_steps, theta = None):

        is_coarse_step_list = [(S_paths.level >0 and (step_i % self.M == self.M - 1) and step_i > 0)
                               for step_i in range(n_steps)]
        expiry = option.expiry

        #print(is_coarse_step_list)
        option_is_path_dependent = option.is_path_dependent
        is_layered_sim = S_paths.level > 0
        dt = expiry / n_steps
        sqrt_dt = math.sqrt(dt)
        dt_sub = self.M * dt

        n_assets = self.assets.n_assets
        n_sims = S_paths.n_sims
        dim = [n_sims, n_assets]
        n_slice = S_paths.n_slice

        if is_layered_sim:
            dW_t_sub = tc.full(S_paths.dim_sub, 0.0, device=device0)


            #S_paths.dW  = [None] * n_steps
        #
        #dW_t = tc.empty(dim, device= device0)
        for step_i in range(n_steps):
            dW_t = S_paths.rnd_normal_dW(step_i, dt)

            if self.correlate_samples:
                dW_t = tc.matmul(dW_t, self.corr_cholmat_R)

            S_t0 = S_paths.S_t
            S_t1 = self.disc_step(S_t0, dW_t, dt)
            S_paths.S_t = S_t1
            S_paths.t += dt
            S_paths.step_i = step_i + 1

            if is_layered_sim:
                dW_t_sub += dW_t[:n_slice]
                if is_coarse_step_list[step_i]:
                    S_t0_sub = S_paths.S_t_sub
                    S_t1_sub = self.disc_step_sub(S_t0_sub, dW_t_sub, dt_sub)
                    S_paths.S_t_sub = S_t1_sub
                    S_paths.t_sub += dt_sub
                    S_paths.step_i_sub = int(step_i/self.M) + 1
                    dW_t_sub.fill_(0.0)
            if option_is_path_dependent:
                option.update_paths(S_paths, sub = is_coarse_step_list[step_i])

        return S_paths
    # [225,  46,  29,  17,  10,   6,   4]


class Heston_Integrator(Integrator):
    def __init__(self, assets, M=2, ImportanceSampling=False):
        super().__init__(assets, M=2, ImportanceSampling=ImportanceSampling)

        self.disc_step_assets = self.assets.Milstein_step_assets
        self.disc_step_vola = self.assets.Milstein_step_vola

        self.log_Heston = False

        if self.log_Heston:
            self.log_disc_step_assets = self.assets.Milstein_step_assets1
            self.log_disc_step_vola = self.assets.Milstein_step_vola1



    def sim_paths_layered(self, S_paths, option, n_steps, theta=None):
        is_coarse_step_list = [((step_i % self.M == self.M - 1) and step_i > 0) for step_i in range(n_steps)]
        expiry = option.expiry

        #self.sampler.soft_reset(n_steps, self.M)

        option_is_path_dependent = option.is_path_dependent
        is_layered_sim = S_paths.level > 0
        dt = expiry / n_steps
        sqrt_dt = math.sqrt(dt)
        dt_sub = self.M * dt

        n_assets = self.assets.n_assets
        n_sims = S_paths.n_sims
        dim = [n_sims, n_assets]
        dim = S_paths.dim

        rho =self.assets.rho
        sqrt_one_minus_rhosq = self.assets.sqrt_one_minus_rhosq

        # Importance_Sampling = True
        if False and theta == None:
            # Importance_Sampling = False
            #theta = tc.tensor([[0] * n_assets] * n_steps, device=device0)
            S_paths.is_importance_sampled = False
            voltheta = tc.tensor([0.0] * n_steps, device=device0)
        else:
            pass
            #voltheta = tc.tensor([-0.05] * n_steps, device=device0)
            S_paths.is_importance_sampled = True
            #if theta.shape[0] < n_steps:
                #theta = tc.tensor([list(theta[0])] * n_steps, device=device0)
        if is_layered_sim:
            dW_t_sub = tc.full(S_paths.dim_sub, 0.0, device=device0)
            dWv_t_sub = tc.full(S_paths.dim_sub, 0.0, device=device0)

        S_paths.theta123 = theta

        if self.ImportanceSampling_training:
            pass
            #S_paths.dW = tc.empty([n_steps, S_paths.n_slice, n_assets], device=device0)
            # S_paths.dW  = [None] * n_steps
        #
        #self.log_Heston = True
        log_Heston = self.log_Heston
        if log_Heston:
            S_paths.S_t = tc.log(S_paths.S_t)
            if is_layered_sim:
                S_paths.S_t_sub = tc.log(S_paths.S_t_sub)
        for step_i in range(n_steps):

            dW_t, dWv_t = S_paths.rnd_normal_dWs_dWv(step_i, dt)

            if self.correlate_samples:
                dW_t = tc.matmul(dW_t, self.corr_cholmat_R)

            v_t0 = tc.abs(S_paths.v_t)
            # full truncation scheme
            #v_t0 = tc.clamp(S_paths.v_t, min = 0)
            S_paths.v_t = v_t0
            sqrt_v_t0 = tc.sqrt(v_t0)

            S_t0 = S_paths.S_t
            if log_Heston:
                S_t1 = self.log_disc_step_assets(S_t0, dW_t, dWv_t, v_t0, sqrt_v_t0, dt)
            else:
                S_t1 = self.disc_step_assets( S_t0, dW_t, v_t0, sqrt_v_t0, dt)
            S_paths.S_t = S_t1
            S_paths.t += dt

            if log_Heston:
                v_t1 = self.log_disc_step_vola(v_t0, sqrt_v_t0, dWv_t, dt)
            else:
                v_t1 = self.disc_step_vola(v_t0, sqrt_v_t0, dWv_t, dt)
            S_paths.v_t = v_t1

            if is_layered_sim:
                dW_t_sub += dW_t[:S_paths.n_slice]
                dWv_t_sub += dWv_t[:S_paths.n_slice]
                if is_coarse_step_list[step_i]:
                    v_t0_sub = tc.abs(S_paths.v_t_sub)
                    sqrt_v_t0_sub = tc.sqrt(v_t0_sub)

                    S_t0_sub = S_paths.S_t_sub
                    if log_Heston:
                        S_t1_sub = self.log_disc_step_assets(S_t0_sub, dW_t_sub, dWv_t_sub, v_t0_sub, sqrt_v_t0_sub, dt_sub)
                    else:
                        S_t1_sub = self.disc_step_assets(S_t0_sub, dW_t_sub, v_t0_sub, sqrt_v_t0_sub, dt_sub)


                    S_paths.S_t_sub = S_t1_sub


                    if log_Heston:
                        v_t1_sub = self.log_disc_step_vola(v_t0_sub, sqrt_v_t0_sub, dWv_t_sub, dt_sub)
                    else:
                        v_t1_sub =  self.disc_step_vola(v_t0_sub, sqrt_v_t0_sub, dWv_t_sub, dt_sub)

                    S_paths.v_t_sub = v_t1_sub

                    S_paths.t_sub += dt_sub
                    dW_t_sub.fill_(0.0)
                    dWv_t_sub.fill_(0.0)
            if option_is_path_dependent:
                option.update_paths(S_paths, sub=is_coarse_step_list[step_i])
        #print(S_paths.S_t[:4])
        if log_Heston:
            S_paths.S_t = tc.exp(S_paths.S_t)
            if is_layered_sim:
                S_paths.S_t_sub = tc.exp(S_paths.S_t_sub)

        return S_paths













