#!/usr/bin/env python3

import numpy as np 


class Neuron:

 
 def __init__(self,nx):

        if not isinstance (nx,int):
            raise TypeError ('nx must be an integer')

        if nx < 1:
            raise ValueError ('nx must be a positive integer')
        
        self.W = np.random.normal(0, 1, (1 ,nx))
        self.b = 0
        self.A = 0

 @property
 def W(self):
     return self.__W

@property
 def b(self):
     return self.__b

@property
 def A(self):
     return self.__A


 def forward_prop(self, X):
    a1 = np.matmul(self.__W, X) + self.__b
    self.__A = 1 /(1 + np.exp(-a1))
    return self.__A

 def cost(self, Y, A):
    m = Y.shape[1]
    cost =  Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
    sigma = np.sum(cost)
    cost =  -(1 / m)*sigma
    return cost
 
 def evaluate(self, X, Y):
    self.forward_prop(X)
    predict = np.where(self.__A >= 0.5, 1, 0)
    cost = self.cost(Y, self.__A)
    return predict, cost

 def gradient_descent(self, X, Y, A, alpha=0.05):
        
        dz = A - Y
        dw = np.matmul(X, dz.T) / A.shape[1]
        db = np.sum(dz) / A.shape[1]
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

 def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True and iterations % step == 0:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        
        return self.evaluate(X, Y)
