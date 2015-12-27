# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from scipy.optimize import fsolve

from RungeKutta import *

class DiagonalImplicitRungeKutta(RungeKutta):	
	def _compute_increments(self, y, t, h, *args):
		nu = self.rk_matrix.shape[0]
		increments = zeros((self.initial_value.size,nu)) 
		
		for i in xrange(nu):
			increment = dot(increments[:,0:i], self.rk_matrix[i,0:i]) 
			
			def func(x, increment):
				arg1 = t + h * self.rk_nodes[i]
				arg2 = y + h * increment + h * self.rk_matrix[i,i] * x
				return self.function(arg1, arg2, *args)  - x
				
			increments[:,i] = fsolve(func, y, increment)
		return increments
