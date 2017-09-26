# MDP Solver

## MDP solving program that uses a variety of different algortihms. Currently supports:
* Value Iteration (VI)
* Policy Iteration (PI)
* Modified Policy Iteration (MPI)
* Linear Program Formulation & Solution (using gurobi) (LP)
* TD(0) Policy Evaluation (TD0)
* Every-Visit Monte-Carlo Policy Evaluation (EVMC)

### Command Line arguments:
* -f, -file: The file to read from (default = 'mdp.csv')
* -m, -method: The method to use. Choose from the above abbreviations (default = 'VI')
* -g, -gamma: The discount value (default = 0.9)
* -a, -alpha: The alpha value to use (e.g for EVMC) (default = 0.1)
* -i, -iterations: The number of iterations for the algorithm (default = 1)
* -r, -runs: The number of runs (total = runs*iterations) (default = 1)
* -t, -threshold: The threshold for VI, PI etc. (default = 0.0001)