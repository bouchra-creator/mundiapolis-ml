#!/usr/bin/env python3

import numpy as np 

class Neuron:
 '''Class Neuron that defines a single neuron performing binary classification '''
 
 
 def __init__(self, nx):
  '''class constructor'''
  
        if not isinstance(nx, int):
            raise TypeError ('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        else: 
            loc, scale = 0, 1
            self.W=np.random.normal(loc, scale, (1, nx))
            self.b=0
            self.A=0
