
import torch as tc


class option():
    def __init__(self, expiry):
        #self.assets = assets
        self.expiry = expiry
        self.is_path_dependent = False


    def init_paths(self):
        pass

    def prepare_paths(self, S_paths):
        pass
        #S_Paths.S_extras1 = [None] * 2
        #S_Paths.S_extras2 = [None] * 2

    def update_paths(self, S_paths):
        pass

    def reset_paths(self, S_paths):
        pass

    def payoff(self, S_t, S_extra):
        return S_t

    def get_payoff(self, S_paths, sub = False):
        if sub:
            return self.payoff(S_t_sub, S_extra_sub)
        else:
            return self.payoff(S_t, S_extra)



class vanilla_option(option):
    def __init__(self, strike, expiry, payofftype = "Call"):
        super().__init__(expiry)
        self.strike = strike
        self.K = strike
        self.expiry = expiry

        assert payofftype in ["Call", "Put"]
        self.payofftype = payofftype
        if self.payofftype == "Call":
            self.payofftype_sign = 1
        elif self.payofftype == "Put":
            self.payofftype_sign = -1

    def payoff(self, S_paths, sub = False, slicing = slice(None)):
        s = slicing
        #S_t = S_paths.S_t[s] if not sub else S_paths.S_t_sub[s]
        S_t = S_paths.S_t if not sub else S_paths.S_t_sub
        return tc.clamp( self.payofftype_sign * (S_t - self.K), min = 0 )


class spread_option(vanilla_option):
    def __init__(self, strike, expiry, payofftype="Call"):
        super().__init__(strike, expiry, payofftype=payofftype)
        self.n_assets_req = 2

    def payoff(self, S_paths, sub = False, slicing = slice(None)):
        s = slicing
        S_t = S_paths.S_t if not sub else S_paths.S_t_sub
        #print(S_t[0], "S_t[0]")
        return tc.clamp(self.payofftype_sign * (S_t[:,0] - S_t[:,1]  - self.K), min = 0 )[:,None]

    def payoff213(self, S_t, S_extra = None):
        #print(S_t[0], "S_t[0]")
        return tc.clamp(self.payofftype_sign * (S_t[:,0] - S_t[:,1]  - self.K), min = 0 )[:,None]

class barrier_option(vanilla_option):
    def __init__(self, strike, barrier, expiry, payofftype="Call", knockouttype = "Knock-Out"):
        super().__init__(strike, expiry, payofftype=payofftype)
        self.is_path_dependent = True
        self.barrier = barrier

        assert knockouttype in ["Knock-Out", "Knock-In"]
        self.knockouttype = knockouttype
        if self.knockouttype == "Knock-Out":
            self.knockouttype_sign = -1
        elif self.knockouttype == "Knock-In":
            self.knockouttype_sign = 1

    def prepare_paths(self, S_paths):
        S_paths.prod_p = tc.empty([S_paths.n_sims, S_paths.n_assets], device=S_paths.device)
        S_paths.S_t_prev = tc.empty([S_paths.n_sims, S_paths.n_assets], device=S_paths.device)
        if S_paths.level > 0:
            S_paths.prod_p_sub = tc.empty([S_paths.n_sims, S_paths.n_assets], device=S_paths.device)
            S_paths.S_t_prev_sub = tc.empty([S_paths.n_sims, S_paths.n_assets], device=S_paths.device)
        else:
            S_paths.prod_p_sub = tc.empty([1, S_paths.n_assets], device=S_paths.device)
            S_paths.S_t_prev_sub = tc.empty([1, S_paths.n_assets], device=S_paths.device)

    def reset_paths(self, S_paths):
        S_paths.prod_p.fill_(1.0)
        for i in range(S_paths.n_assets):
            S_paths.S_t_prev[:,i].fill_(S_paths.assets.S0[i])
        if S_paths.level > 0:
            S_paths.prod_p_sub.fill_(1.0)
            for i in range(S_paths.n_assets):
                S_paths.S_t_prev_sub[:, i].fill_(S_paths.assets.S0[i])



    def update_paths(self, S_paths, sub = False):
        l = S_paths.level
        dt = S_paths.calling_mcalgo.dt[l]
        dt_sub = S_paths.calling_mcalgo.dt_sub[l]
        B = self.barrier
        b = self.knockouttype_sign
        sigma = S_paths.assets.Sigma[0]
        S_t0 = S_paths.S_t_prev
        S_t1 = S_paths.S_t
        p = tc.exp( tc.divide( -2 * tc.clamp(b * (S_t0 - B), min=0) * tc.clamp( b * (S_t1 - B), min=0) ,
                               ((sigma * S_t0) ** 2) * dt) )
        S_paths.prod_p = S_paths.prod_p * (1 - p)
        S_paths.S_t_prev = S_paths.S_t
        if sub:
            S_t0 = S_paths.S_t_prev_sub
            S_t1 = S_paths.S_t_sub
            p = tc.exp(tc.divide(-2 * tc.clamp(b * (S_t0 - B), min=0) * tc.clamp(b * (S_t1 - B), min=0),
                                 ((sigma * S_t0) ** 2) * dt_sub))
            S_paths.prod_p_sub = S_paths.prod_p_sub * (1 - p)
            S_paths.S_t_prev_sub = S_paths.S_t_sub


    def payoff(self, S_paths, sub = False, slicing = slice(None)):
        s = slicing
        S_t = S_paths.S_t[s] if not sub else S_paths.S_t_sub[s]
        prod_p = S_paths.prod_p[s] if not sub else S_paths.prod_p_sub[s]
        #print(prod_p.shape)
        #print(S_t.shape)
        return tc.clamp(self.payofftype_sign * (S_t - self.K), min = 0 ) * prod_p