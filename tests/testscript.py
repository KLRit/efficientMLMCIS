

#['0.662s', '1.340s'] [5.2749683380126955, 21.459500217437743]
#['3.140s', '5.897s'] [19.31468257904053, 72.81048774719238]
import time
from testcases import *
from testfunc import *

#hide_prints = False
hide_prints = True
includeAT = True
#includeAT = False

#global epsilon to test MLMCIS, MLMC against
geps = 0.01
n_tries  = 5
start = time.time()

#MonteCarlo(p1c1_benchOP)

# values other than fixed_L = None fix the number of levels to compare variance reductions
fixed_L = None
#Vanilla call
#test_run(p1a_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test barrier option
test_run(p1c1_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, includeAT= includeAT, fixed_L = fixed_L)
#test heston call option
test_run(p4a_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, includeAT= includeAT, fixed_L = fixed_L)
#test Spread-call option on 2 assets
test_run(p6a1_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, includeAT= includeAT, fixed_L= fixed_L)
end = time.time()
print(end -start, "s")
quit()
#test_run(p1c2_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, includeAT= includeAT, fixed_L = fixed_L)
#test_run(p1c3_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p1a_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, fixed_L = fixed_L)

test_run(p6a1_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, includeAT= includeAT, fixed_L= fixed_L)
#test_run(p6a3_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, includeAT= True, fixed_L= fixed_L)
test_run(p4a_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, fixed_L = fixed_L)
#test_run(p4a2_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, fixed_L = fixed_L)


#test_run(p1a_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p1a1_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p1a2_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p1a2c_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p1c1_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p1c2c_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p1c2_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p4a_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)
#test_run(p6a1_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints, includeAT= False)
#test_run(p6a3_benchOP, eps = geps, n_tries = n_tries, hide_prints= hide_prints)

#test_run(p1a_benchOP)

epsilons =[0.01, 0.005, 0.002]
#epsilons =[0.01, 0.005] #, 0.002]
#plot1 = plot_algos(p4a_benchOP,  epsilons, n_tries = 1, MLMConly = True, fixed_L = None)
#plot1 = plot_algos(p1c1_benchOP,  epsilons, n_tries = 5, MLMConly = True, fixed_L = None)
#plot_algos(p1c1_benchOP, epsilons, n_tries = 1, MLMConly = True)

plot1.show()

