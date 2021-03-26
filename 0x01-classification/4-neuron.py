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
    cost = - Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
    sigma = np.sum(cost)
    m = Y.shape[1]
    cost =  -1/(sigma/m)
    return cost
 
def evaluate(self, X, Y):
    self.forward_prop(X)
    predict = np.where(self.__A >= 0.5, 1, 0)
    cost = self.cost(Y, self.__A)
    return predict, cost
  
  
