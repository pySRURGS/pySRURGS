![Binoculars](image/Gnome-system-search.jpg)

# pySRURGS - Symbolic Regression by Uniform Random Global Search (in python)

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
7. Developed and tested on Python3.6

## Getting Started

It's a python3 script. Just download it and run it via a terminal.

### Prerequisites

You can install the prerequisite packages with pip

```mpmath lmfit sympy pandas numpy parmap tqdm sqlitedict tabulate```

### Installing

Install the prerequisites then download the script.

```
pip install mpmath lmfit sympy pandas numpy parmap tqdm sqlitedict tabulate
git clone https://github.com/pySRURGS/pySRURGS.git
```

### Command line help

```
python3 pySRURGS.py -h
```

The above command should render the following:

```
pySRURGS - Symbolic Regression by Uniform Random Global Search
Sohrab Towfighi (C) 2019
License: GPL 3.0

All your data needs to be numeric.
Your CSV file should have a header.
Inside the csv, the dependent variable should be the rightmost column.
Do not use special characters or spaces in variable names.

The config.py file defines the number of fitting parameters, the number of
permitted binary trees through which we search, and the types of functions
permitted in the search space.

usage: pySRURGS.py [-h] [-test TEST] [-run_ID RUN_ID] [-single] [-count]
                   train iters

positional arguments:
  train           absolute or relative file path to the csv file housing the
                  training data
  iters           the number of equations to be attempted in this run

optional arguments:
  -h, --help      show this help message and exit
  -test TEST      absolute or relative file path to the csv file housing the
                  testing data
  -run_ID RUN_ID  some text that uniquely identifies this run
  -single         run in single processing mode
  -count          Prints out how many possible equations for this
                  configuration. No other processing performed.
```

### An example

A sample problem is provided. The filename denotes the true equation.

```
$ winpty python pySRURGS.py ./csvs/x1_squared_minus_five_x3.csv 10 2
Running in single processor mode
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:27<00:00,  2.74s/it]
  Normalized Mean Squared Error        R^2  Equation, simplified                                                                                           Parameters
-------------------------------  ---------  -------------------------------------------------------------------------------------------------------------  ----------------------------------------------
                        10.2672   0.963167  -(x0 - x1*((p1*p3 - x1*x3)/(p1*x1))**x3*(p3 - x0))*(p0*x1*(x0*(x1 + x3) - x1) - x0*sin(p2))/(x0*x1*(p3 - x0))  -4.74E-04,1.51E+04,2.14E-01,5.25E+03,-1.27E+04
                        17.338    0.941908  x2 + p4*x3/p1**2 - p0**(-x1)*p4*x5*log(p0)/x4 - p0**(-x1)*x2*x5*log(p0)/x4                                     1.00E+00,1.69E+02,1.00E+00,1.00E+00,1.33E+05
                        65.4971   0.742457  (2*p3 - p4**x2)*(-p0*p3*x0 + p0*x2 + x0*x2)*((-p2 + p3)**(x3/p0))**(-p1 + x2)/(p0*x0*sin(p1))                  1.88E+00,-1.15E+00,-3.12E+00,4.76E-01,1.07E-01
                       111.863    0.461941  x5**(-x4)*(-x3**x5*(p4**p0 - 2*x5))**sinh(p0)*(p0*x5 - p1*x0)/(p1*(p2 - x1))                                   8.75E-01,5.85E-04,1.00E+00,1.00E+00,5.76E+00
                       256.675   -1.05228   (x1 + x2)*(x2 + x4)*(p0 - p0**p1 + x2)*(p0 - x0 + x3 + x5)*log(p0)                                             1.03E+00,1.03E+00,1.00E+00,1.00E+00,1.00E+00
```

### Configuring the search

Inside config.py, you will find the definition of the permitted functions. The list elements need to be the function names input as strings. The *f_functions* list is for the functions of arity 1. The *n_functions* list is for functions of arity 2. You can change the maximum number of permitted fitting parameters.

## Author

**Sohrab Towfighi**

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Luther Tychonievich created the algorithm mapping integers to full binary trees
* The icon is from the GNOME icon project and the respective artists.
