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
4. Loads data from csv
5. Results are saved to file. New runs are added to previously saved results.

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
pySRURGS - Symbolic Regression by Uniform Random Global Search (in python!)
Sohrab Towfighi (C) 2019
License: GPL 3.0

All your data needs to be numeric. 
Your CSV file should have a header.
Inside the csv, the dependent variable should be the rightmost column.
Do not use special characters or spaces in variable names.

USAGE:
pySRURGS.py $path_to_csv $max_num_evals

ARGUMENTS
1. path_to_csv - An absolute or relative file path to the csv file.
2. max_num_evals - An integer: The number of equations which will be considered in the search.
```

### An example

A sample problem is provided. The filename denotes the true equation.

```
python pySRURGS.py ./x1_squared_minus_five_x3.csv 1000

1008it [04:13,  4.47it/s]
  Mean Squared Error       R^2  Equation, simplified
--------------------  --------  --------------------------------------------------------------------------
             2.60238  0.994864  ((p0 + p2*(-x1 + x4))/p2)**(p1 + x3**2)*((-p0*x3*x4 + p2)/(p0*x4))**(1/p1)
            11.8229   0.969479  (p1*(x1 + 1)/x1)**p3*(p3 + x1**(p1 + x5))**(x1*(p3 - x4))*(p0*x1 + x3)
            13.5942   0.970111  (p0 + p2*(p2 + (p1*(p3 + x2)**x0)**(p2 - x1 + x2)) + x3*(p3 - x0))/p2
            14.6246   0.96641   ((p3*(p2 + x3)**p0)**(p3 + x0 + x1))**(p0*x4*x5 + p1*x2*(p1 + x0))
            14.7199   0.976367  (x0**x0 - x3)*(-p0 + x1 + x1**(2*p0))
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
