# Dholes Hunting-based Optimization: A Novel Algorithm for Global Search and Its Application

## Install Python Lib
```sh 
conda create -n new_env python==3.7.5
conda activate new_env
pip install -r requirements.txt
```

## How to run code?
### Run benchmark
```sh
usage: run_benchmark.py [-h] [-f FUNCTIONS] [-d DIM]
                        [-a ALGORITHMS] [-r RUN]
                        [-z PROBLEM_SIZE] [-e EPOCH_BOUND]
                        [-p POP_SIZE] [-o OUTPUT] [-j JOBS]
                        [-m MULTIPROCESSING]
                        [-t TIME_BOUND] [-vb verbose]
                        [-n N_TRIALS] [-x MODE]
```
```sh
optional arguments:
  -h, --help            show this help message and exit
  -f FUNCTIONS, --functions FUNCTIONS
                        list of benchmark functions
  -d DIM, --dim DIM     number of dimensions
  -a ALGORITHMS, --algorithms ALGORITHMS
                        list of test algorithms, default all
  -r RUN, --run RUN     number of run times
  -z PROBLEM_SIZE, --problem_size PROBLEM_SIZE
                        size of problem
  -e EPOCH_BOUND, --epoch_bound EPOCH_BOUND
                        number of iterations
  -p POP_SIZE, --pop-size POP_SIZE
                        population size
  -o OUTPUT, --output OUTPUT
                        output path
  -j JOBS, --jobs JOBS  number of parallel processes
  -m MULTIPROCESSING, --multiprocessing MULTIPROCESSING
                        Run on multiprocessing
  -t TIME_BOUND, --time_bound TIME_BOUND
                        Time bound for trainning (s)
  -vb verbose, --verbose verbose
                        log fitness by epoch
  -n N_TRIALS, --n_trials N_TRIALS
                        number of trials
  -x MODE, --mode MODE  run with the time bound or epoch
                        bound
```

### Run Application
```sh
usage: run_application.py [-h] [-d DIM] [-a ALGORITHMS] [-r RUN]
                          [-z PROBLEM_SIZE] [-e EPOCH_BOUND] [-p POP_SIZE]
                          [-o OUTPUT] [-j JOBS] [-m MULTIPROCESSING]
                          [-t TIME_BOUND] [-v verbose] [-n N_TRIALS] [-x MODE]
                          [-q N_VALUES] [-s NUM_SIM]
```
```sh

optional arguments:
  -h, --help            show this help message and exit
  -d DIM, --dim DIM     number of dimensions
  -a ALGORITHMS, --algorithms ALGORITHMS
                        list of test algorithms
  -r RUN, --run RUN     number of run times
  -z PROBLEM_SIZE, --problem_size PROBLEM_SIZE
                        size of problem
  -e EPOCH_BOUND, --epoch_bound EPOCH_BOUND
                        number of iterations
  -p POP_SIZE, --pop-size POP_SIZE
                        population size
  -o OUTPUT, --output OUTPUT
                        output path
  -j JOBS, --jobs JOBS  number of parallel processes
  -m MULTIPROCESSING, --multiprocessing MULTIPROCESSING
                        Run on multiprocessing
  -t TIME_BOUND, --time_bound TIME_BOUND
                        Time bound for trainning (s)
  -v verbose, --verbose verbose
                        log fitness by epoch
  -n N_TRIALS, --n_trials N_TRIALS
                        number of trials
  -x MODE, --mode MODE  run with the time bound or epoch bound
  -q N_VALUES, --n_values N_VALUES
                        number of values for scenario
  -s NUM_SIM, --num_sim NUM_SIM
                        number of simulation for each solution
```
`The results stored in the output folder`

## Contact

* If you want to know more about code, or want a pdf of both above paper, contact hienvq.2000@gmail.com or nguyenthieu2102@gmail.com or my 

* Take a look at this repos, the comprehensive metaheuristics library

    * https://github.com/thieu1995/mealpy

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The source code for the site is licensed under the MIT license, which you can find in the LICENSE file.
