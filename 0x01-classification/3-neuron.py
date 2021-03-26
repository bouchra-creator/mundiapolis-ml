#!/usr/bin/env python3

import numpy as np 
class Neuron:

 def __init__(self,nx):

        if not isinstance (nx,int):
            raise TypeError ('nx must be an integer')

        if nx < 1:
            raise ValueError ('nx must be a positive integer')
            self.W=np.random.normal(0,1,(1,nx))
            self.b=0
            self.A=0
@property
 def get_W(self):
     return self.__W

@property
def get_b(self):
     return self.__b

@property
def get_A(self):
     return self.__A

@property   
def get_W(self,w):
      self.__W = W

@property
def get_b(self,b):
      self.__b = b

@property
def get_A(self,A):
      self.__A = A

def forward_prop(self, X):
    a1=np.matmul(self.__W, X) + self.__b
    self.__A = 1 /(1 + np.exp(-a1))
    return self.__A

def cost(self, Y, A):
    cost = - Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
    sigma = np.sum(cost)
    m = Y.shape[1]
    cost =  -1/(sigma/m)
    return cost

