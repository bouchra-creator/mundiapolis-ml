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
        
   def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        n1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-n1))
        n2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-n2))
        return self.__A1, self.__A2
      
    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost
      
     def evaluate(self, X, Y):
        """
        evaluates the neural network prediction
        """
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return prediction, cost
        
     def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        calculates one pass of gradient descent on the neuron
        """
       
        da2 = A2 - Y
        dw2 = np.matmul(A1, da2.T) / A1.shape[1]
        db2 = np.sum(dz2, 1, True) / A2.shape[1]

        da1 = A1 * (1 - A1)
        da1 = np.matmul(self.__W2.T, da2)
        da1 = da1 * da1
        dw1 = np.matmul(X, da1.T) / A1.shape[1]
        db1 = np.sum(da1, 1, True) / A1.shape[1]
        self.__W2 = self.__W2 - alpha * dw2.T
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1.T
        
     @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
