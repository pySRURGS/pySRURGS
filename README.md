![Binoculars](image/Gnome-system-search.png)

# pySRURGS - Symbolic Regression by Uniform Random Global Search (in python!)

Symbolic regression is a type of data analysis problem where you search for the 
equation of best fit for a numerical dataset. This package does this task by 
randomly, with uniform probability of selection, guessing candidate solutions 
and evaluating them. The No Free Lunch Theorem argues that this approach should 
be equivalent to other approaches like Genetic Programming. It is probably faster 
because of less overhead. 

## Features 

1. Robust parameter fitting
2. Multiprocessing for faster computing
3. Load data from csv

## Getting Started

It's a python3 script. Just download it and run it via a terminal.

### Prerequisites

You can install the prerequisite packages with pip

```
mpmath lmfit sympy pandas numpy
```

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


```
pySRURGS - Symbolic Regression by Uniform Random Global Search (in python!)
Sohrab Towfighi (C) 2019
Licence: GPL 3.0

All your data needs to be numeric. Your CSV file should have a header.

USAGE:
pySRURGS.py $path_to_csv $max_number_fitting_params $max_num_evals
   
path_to_csv: an absolute or relative file path to the csv file, the dependent 
             variable should be the rightmost variable
max_number_fitting_params: an integer. The fewer of these you have, the fewer 
                           fitting constants will be permitted.
max_number_equations_attempted: an integer. The greater this value, the more 
                                time your computations will take.

```

### An example

The code takes several arguments as inputs

```
Give an example
```


## Authors

**Sohrab Towfighi**

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Luther Tychonievich created the algorithm mapping integers to full binary trees
* The icon is from the GNOME icon project and the respective artists.
