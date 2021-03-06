#!/usr/bin/env python3

import numpy as np 


class NeuralNetwork:

 
def __init__(self, nx, nodes):

       if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(nx, nodes) :
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
            
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0
