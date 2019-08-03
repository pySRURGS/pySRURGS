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

### An example

The code takes several arguments as inputs

```
Give an example
```


## Authors

**Sohrab Towfighi**

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Luther Tychonievich created the algorithm mapping integers to full binary trees
