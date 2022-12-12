################# VALIDATION ##############################

import os, sys
import datetime
import time
import math
import torch as tc
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np



print("Torch VersioN", tc.version.__version__)

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'src'))
from assets import *
from options import *
from mcalgo import *
from testcases import *

print("asdasdasd")

geps = 0.002 #0.000005
n_tries_default = 2
notcrude = True
print(datetime.datetime.now())

tc.manual_seed(time.time())


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Nothing:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def test_run( option_on_assets, eps = 0.01, tolerance = 2, n_tries = n_tries_default,
              MLMConly = True, hide_prints = False, includeAT = True,
              fixed_L = None):
    print("testrun")
    option = option_on_assets.option
    assets = option_on_assets.assets
    exact_price = option_on_assets.exact_price
    test_run.test1 = 123
    if hide_prints:
        print_class = HiddenPrints
        #print_class = Nothing
    else:
        print_class = Nothing

    def mse(error):
        return np.mean(np.square(errors))

    def vse(error):
        return np.var(np.square(errors))

    def rmse(error):
        return np.sqrt(mse(error))

    def mae(error):
        return np.mean(np.abs(error))


    vars = []
    bias_sq = []

    #n_tries = 5
    mean_estimates = []
    mean_errors = []
    mean_errsq_times = []
    mean_abs_errors = []
    mean_sq_errors = []
    var_sq_errors = []
    stds_errors = []
    rms_errors = []
    E_vars = []
    mean_vars = []
    stds_vars = []
    mean_total_times = []
    stds_total_times = []
    mean_sim_times = []
    stds_sim_times = []
    mean_prep_times = []
    stds_prep_times = []
    mean_Costs = []

    crude_n_steps = 128
    #for i in range(1):

    do_a_flush_run = False
    QRS = False #True
    includeQMLMC = False #True
    algos = []
    algonames = []

    #includeAT = False
    if includeAT:
       algos += [MultiLevelMonteCarlo(option, assets, error_epsilon = eps,  ImportanceSampling = True, AntitheticSampling = True, fixed_L = fixed_L ),
               MultiLevelMonteCarlo(option, assets, error_epsilon = eps,  AntitheticSampling = True, fixed_L = fixed_L )]
       algonames += ["MLMC-ATIS", "MLMC-ATS"]

    #algos += [MultiLevelMonteCarlo(option1, eps = eps,  ImportanceSampling = True),
    #             MultiLevelMonteCarlo(option1, eps = eps)]

    algos += [MultiLevelMonteCarlo(option, assets, error_epsilon = eps, ImportanceSampling = True, fixed_L = fixed_L),
              MultiLevelMonteCarlo(option, assets, error_epsilon = eps, ImportanceSampling = False, fixed_L = fixed_L)]

    algonames += ["MLMC-IS", "MLMC"]

    #algos = [MultiLevelMonteCarlo(option1, eps = eps, Importance_Sampling = True, Quasirandom_Sampling = QRS ),
    #       MultiLevelMonteCarlo(option1, eps = eps, Quasirandom_Sampling = QRS )]

    # algos = [MultiLevelMonteCarlo(option1, eps = eps, Importance_Sampling = True ),
    #          MultiLevelMonteCarlo(option1, eps = eps)]
    if not MLMConly:
        algos += [MonteCarlo(option, assets),
             MonteCarlo(option, assets)]
        algonames += ["CrudeMCIS", "CrudeMC"]

    ji = 0
    for algo in algos:


        print(algonames[ji])
        if do_a_flush_run:
            with HiddenPrints():
                algo.sim()
                algo.reset()

        estimates = []
        errors = []
        vars = []
        Costs = []
        total_times = []
        sim_times = []
        prep_times = []

        errsq_times = []
        #print(n_tries, "$$$$$$$$$$$$$$$$$$$$$$$$")
        for i in range(n_tries):
            algo.reset()
            tc.random.manual_seed(7 * time.time() + 17)       # local pytorch random calls are BROKEN!!!
            #MLMCIS = MultiLevelMonteCarlo(option1, eps = eps)
            with print_class():
            #with HiddenPrints():
                Res, Var = algo.sim()

            #print(algo.level_var)
            #print(algo.n_batches_req)
            #print(algo.level_batch_time[:(algo.L + 1)])
            #print("&&&&&&&&&&&& RES", Res)
            Costs.append(algo.C.cpu())
            estimates.append(Res)
            err = (Res - exact_price)
            #err = abs(Res - exact_price)
            #print("&&&&&&&&&&&& err", err)
            errors.append(err)
            vars.append(Var)
            total_times.append(algo.time_total)
            errsq_times.append((err ** 2) * algo.time_total)
            sim_times.append(algo.time_sim)
            prep_times.append(algo.time_prep)
        print("errors", errors)
        print("rms", math.sqrt((1/ n_tries) * sum([err ** 2  for err in errors]) ))
        print("total times", total_times)
        print("total times level", algo.level_time_sum)
        print("prep times level", algo.level_time_prep)
        if algo.asset_coll.n_assets < 2:
            print("Theta0", [float(The) for The in algo.Theta[:algo.L+1]])
        else:
            print("Theta0", [The for The in algo.Theta[:algo.L + 1]])
        print("dTheta0", [The for The in algo.dTheta[:algo.L + 1]])
        print("convergence params", algo.estimate_convergence_params())
        print(algo.level_time_sum[:algo.L+1])
        print("batch time", algo.level_batch_time_running[:algo.L + 1])
        print(algo.n_batches_done[:algo.L+1])
        #print(algo.Theta[:algo.L+1], "Theta 0")
        print("level res", algo.level_res[:algo.L+1])
        print("level var", algo.level_var[:algo.L+1])
        print(algo.C)

        E_vars.append(np.var(estimates))
        bias_sq.append(np.square(np.mean(np.array(estimates) - exact_price)))


        mean_total_times.append(np.mean(total_times))
        mean_errsq_times.append( (10 ** 6) * float(np.mean(errsq_times)))
        stds_total_times.append(np.std(total_times))
        mean_sim_times.append(np.mean(sim_times))
        stds_sim_times.append(np.std(sim_times))
        mean_prep_times.append(np.mean(prep_times))
        stds_prep_times.append(np.std(prep_times))
        mean_errors.append(np.mean(errors))
        mean_estimates.append(np.mean(estimates))
        mean_sq_errors.append(mse(errors))
        var_sq_errors.append(vse(errors))
        mean_abs_errors.append(mae(errors))
        stds_errors.append(np.std(errors))
        rms_errors.append(rmse(errors))
        mean_vars.append(np.mean(vars))
        stds_vars.append(np.std(vars))
        mean_Costs.append(np.mean(Costs))
        #print("sims", ["{:.2f}".format(x) for x in np.divide(algo.n_sims_total, 10 ** 6).tolist()] )
        #print("time", ["{:.3f}".format(x) for x in algo.time_per_level])
        #print(float(np.sum(algo.time_per_level)))
        #print(algo.Theta, "Teta 0")
        #algo.approx_convergence_params()
        ji = ji + 1
    #print(total_times)
    mean_errors = np.array(np.abs(mean_errors))
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print(option_on_assets.description)
    print("exact price ", exact_price)
    print(algonames)
    tests_passed_bools = (mean_errors < (eps * tolerance)).tolist()
    color = "red"
    if all(tests_passed_bools):
        color = "green"
    print(colored(tests_passed_bools, color))
    print([ '%.3f' % elem + "s" for elem in mean_total_times], "mean total time")
    print(['%.3f' % elem + " s" for elem in mean_errsq_times], "mean err^2 * time  * (10 ** 6)")
    print([ '%.3f' % elem + "s" for elem in stds_total_times], "std total time")
    print([ '%.3f' % elem + "s" for elem in mean_sim_times], "mean sim   time")
    print([ '%.3f' % elem + "s" for elem in stds_sim_times], "std sim time")
    print([ '%.3f' % elem + "s" for elem in mean_prep_times], "mean prep  time")
    print([ '%.3f' % elem + "s" for elem in stds_prep_times], "std prep time")
    #print(mean_errors.tolist())
    print(mean_estimates, "mean estimates")
    print(rms_errors, "rms errors")
    print(mean_Costs, "mean C")
    print(stds_errors, "std errors")
    print(mean_abs_errors, "ma errors")
    print(mean_sq_errors, "mse")
    print(np.sqrt(var_sq_errors), "rvse")
    print(var_sq_errors, "vse") #, np.sqrt(var_sq_errors), "rvse")
    print(mean_vars, "rms vars")
    print(stds_vars, "std varss")

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(E_vars, "vars")
    # print(bias_sq, "bias sq")
    var_plus_bias_sq = np.array(E_vars) + np.array(bias_sq)
    # print(var_plus_bias_sq.tolist(), "vars + bias sq")
    # print(np.sqrt(var_plus_bias_sq).tolist(), "root vars + bias sq")
    print((np.array(bias_sq) / var_plus_bias_sq).tolist(), "bias split")
    print((np.array(E_vars) / var_plus_bias_sq).tolist(), "var split")
    # print((np.array(bias_sq) / (eps ** 2) ).tolist(), "bias sq split eps sq")
    # print((np.array(E_vars) / (eps ** 2 )).tolist(), "var split eps sq")

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print(rms_errors, mean_errsq_times)
    return rms_errors, mean_total_times, algonames, mean_errsq_times


def plot_algos(option1, epsilons, n_tries = 10, MLMConly = True, fixed_L = None):
    exact = option1.exact_price
    #hide_pr
    X, Y = [], []
    with HiddenPrints():
        for eps in epsilons:
            #res = test_run(option1, exact, eps, 2, n_tries = n_tries, MLMConly = MLMConly)
            res = test_run(option1, eps = eps, n_tries = n_tries, fixed_L = fixed_L)
            errors, times = res[0], res[1]
            error_sq_times = res[3]
            Y.append(error_sq_times)
            X.append(errors)
            #Y.append(times)

            algonames = res[2]

    fig, ax = plt.subplots()
    #X, Y = np.array(X) ** 2, np.array(Y)
    X, Y = np.array(X) , np.array(Y)

    colors =["green", "blue", "purple", "red"]
    for i in range(len(errors)):
        print(X[:,i], "errors rms ** 2")
        print(Y[:,i], "times avg")
        #ax.plot(np.log10(X[:,i]), np.log10(Y[:,i]), color = colors[i], label =algonames[i])
        ax.plot(np.log10(X[:, i]), Y[:, i], color=colors[i], label=algonames[i])
    ax.grid()
    #plt.yscale("log")
    #plt.xscale("log")

    return plt

print("test")

