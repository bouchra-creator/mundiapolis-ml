#!/usr/bin/env python3

import numpy as np 


class Neuron:
 '''Class Neuron that defines a single neuron performing binary classification '''
 
  def __init__(self, nx):
     '''class constructor'''
  
        if not isinstance(nx, int):
            raise TypeError ('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        
        self.W =  np.random.normal(1, 0, (1, nx))
        self.b = 0
        self.A = 0
