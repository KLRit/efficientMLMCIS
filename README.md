efficientMLMCIS 
=======
efficient Importance Sampling for Multilevel Monte Carlo
Implemented in Torch for CUDA



**/test/testscript.py**    test option pricing

**/test/testfunc.py**      wrapper for MLMC

**/test/testcases.py**    defines testcases


**/src/mcalgo.py**        main Multilevel loop


**/src/paths.py**          stores paths, handles random number generation, evaluation and optimization

**/src/integrators.py**

**/src/assets.py**        defines asset classes that define discretization schemes

**/src/options.py**         defines option payoffs and updates for path-dependent options





 Sydow et al: [Benchmarking project option pricing](http://www.it.uu.se/research/scientific_computing/project/compfin/benchop)
/ 

A. Kebaier, J. Lelong: [Importance Sampling & SAA](https://arxiv.org/abs/1510.03590) /  

 M. B. Giles: 
[Multilevel Monte Carlo](https://people.maths.ox.ac.uk/gilesm/mlmc.html) /


