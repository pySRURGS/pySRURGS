import sympy
import numpy as np

def sympy_Sub(a, b):
    return sympy.Add(a, -b)


def sympy_Div(a, b):
    return a * sympy.Pow(b, -1)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def tan(x):
    return np.tan(x)


def exp(x):
    return np.exp(x)


def log(x):
    ''' Note, this is actually log(abs(x)) '''
    return np.log(np.abs(x))


def sinh(x):
    return np.sinh(x)


def cosh(x):
    return np.cosh(x)


def tanh(x):
    return np.tanh(x)


def sum(x):
    return np.sum(x)


def add(x, y):
    return np.add(x, y)


def sub(x, y):
    return np.subtract(x, y)


def mul(x, y):
    return np.multiply(x, y)


def div(x, y):
    return np.divide(x, y)


def pow(x, y):
    return np.power(x, y)
