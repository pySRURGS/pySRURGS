
![Binoculars](image/Gnome-system-search.jpg)

# pySRURGS - Symbolic Regression by Uniform Random Global Search (in python)
[![Build Status](https://travis-ci.com/pySRURGS/pySRURGS.svg?branch=master)](https://travis-ci.com/pySRURGS/pySRURGS)
[![GPL Licence](https://badges.frapsoft.com/os/gpl/gpl.svg?v=103)](https://opensource.org/licenses/GPL-3.0/)

Symbolic regression is a type of data analysis problem where you search for the 
equation of best fit for a numerical dataset. This package does this task by 
randomly, with uniform probability of selection, guessing candidate solutions 
and evaluating them. The No Free Lunch Theorem argues that this approach should 
be equivalent to other approaches like Genetic Programming. It is probably faster 
because of less computational overhead. 

## Features 

1. Robust parameter fitting
2. Multiprocessing for faster computing
3. Avoids considering arithmetically equivalent equations
4. Loads data from spreadsheets (comma separated value files)
5. Memoization so that computations speed up after a few iterations 
6. Results are saved to file. New runs are added to previously saved results.
7. User specified number of fitting parameters 
8. User specified number of permitted unique binary trees, which determine the possible equation forms 
9. User specified permitted functions 
10. Developed and tested on Python 3.6

## Getting Started

It's a python3 script. Download it and run it via a terminal.

### Prerequisites

You can install the prerequisite packages with pip

```mpmath lmfit sympy pandas numpy parmap tqdm sqlitedict tabulate```

### Installing

Clone the repo then install the prerequisites.

```
git clone https://github.com/pySRURGS/pySRURGS.git
pip install -r requirements.txt
```

### Command line help

```
python3 pySRURGS.py -h
```

The above command should render the following:

```
usage: pySRURGS.py [-h] [-run_ID RUN_ID] [-single] [-count]
                   [-benchmarks] [-funcs_arity_two FUNCS_ARITY_TWO]
                   [-funcs_arity_one FUNCS_ARITY_ONE]
                   [-max_num_fit_params MAX_NUM_FIT_PARAMS]
                   [-max_size_trees MAX_SIZE_TREES]
                   train iters

positional arguments:
  train                 absolute or relative file path to the csv file housing
                        the training data
  iters                 the number of equations to be attempted in this run

optional arguments:
  -h, --help            show this help message and exit
  -run_ID RUN_ID        some text that uniquely identifies this run (default:
                        None)
  -single               run in single processing mode (default: False)
  -count                Instead of doing symbolic regression, just count out
                        how many possible equations for this configuration. No
                        other processing performed. (default: False)
  -benchmarks           Instead of doing symbolic regression, generate the 100
                        benchmark problems. No other processing performed.
                        (default: False)
  -funcs_arity_two FUNCS_ARITY_TWO
                        a comma separated string listing the functions of
                        arity two you want to be considered.
                        Permitted:add,sub,mul,div,pow (default:
                        add,sub,mul,div,pow)
  -funcs_arity_one FUNCS_ARITY_ONE
                        a comma separated string listing the functions of
                        arity one you want to be considered.
                        Permitted:sin,cos,tan,exp,log,sinh,cosh,tanh (default:
                        None)
  -max_num_fit_params MAX_NUM_FIT_PARAMS
                        the maximum number of fitting parameters permitted in
                        the generated models (default: 3)
  -max_permitted_trees MAX_PERMITTED_TREES
                        the number of unique binary trees that are permitted
                        in the generated models - binary trees define the form
                        of the equation, increasing this number tends to
                        increase the complexity of generated equations
                        (default: 1000)
```

### An example

A sample problem is provided. The filename denotes the true equation.

```

$ winpty python pySRURGS.py -max_num_fit_params 3 -max_permitted_trees 2000 ./csvs/quartic_polynomial.csv 100
Running in multi processor mode
1008it [03:03,  7.78it/s]
  Normalized Mean Squared Error       R^2  Equation, simplified                                                 Parameters
-------------------------------  --------  -------------------------------------------------------------------  ---------------------------
                     0.00518398  0.999871  (p2**p2)**p1*(p2**p2)**(p1*x) + ((p0**2 + p1)**x)**(-p2)             6.16E-01,3.41E+00,1.01E+00
                     0.00623639  0.999845  p0**p2*(p0*x**p2 - p2)**(p1 + 1)*(p1**x)**p1                         3.11E+00,6.34E-01,1.43E+00
                     0.0851371   0.997893  (p0**(p0 + 1) + p1*(p2 + x)**x)**(-p2*x*(p0 - 1)*(p1 + p2 - x))      1.36E+00,1.60E+00,2.85E-01
                     0.108024    0.997381  (p0 + p1*(-p2 + p2**p1 - x + x**(2*p2)) - 1)/p1                      4.89E-01,1.10E+00,7.58E-01
                     0.286993    0.99297   -p2*x**2*(p0 + 1)/(p0*(p2*(p0*x - p0 - p1 + x) + x)*(p0 + p1 + p2))  2.59E+00,6.91E+00,-2.73E-02

```

### Important details 

All your data needs to be numeric.
Your CSV file should have a header.
Inside the csv, the dependent variable should be the rightmost column.
Do not use special characters or spaces in variable names.

## Author

**Sohrab Towfighi**

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Luther Tychonievich created the algorithm mapping integers to full binary trees
* The icon is from the GNOME icon project and the respective artists.
