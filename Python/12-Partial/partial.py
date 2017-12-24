# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):   # x1 = 4.0
    return x**2 + 4.0**2 

def function_2(x):   # x0 = 3.0
    return 3.0**2 + x**2 


print(numerical_diff(function_1,3.0))  #partial x0 and x1 = 3.0
print(numerical_diff(function_2,4.0))  #partial x1 and x0 = 4.0