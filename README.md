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
5. Results are saved to a persistent dictionary for later analysis

## Getting Started

It's a python3 script. Just download it and run it via a terminal.

### Prerequisites

You can install the prerequisite packages with pip

```mpmath lmfit sympy pandas numpy parmap tqdm sqlitedict```

### Installing

Install the prerequisites then download the script.

```
pip install mpmath lmfit sympy pandas numpy
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
pySRURGS.py ./x1_squared_minus_five_x3.csv 100
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
