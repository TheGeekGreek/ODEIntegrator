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
	def _compute_increments(self, y, t, h):
		nu = self.RKMatrix.shape[0]
		increments = zeros((self.initialValues.size,nu)) 
		
		for i in xrange(nu):
			increment = numpy.sum(self.RKMatrix[i,0:i] * increments[:,0:i], axis = 1)
			
			def func(x, increment):
				arg1 = t + h * self.RKNodes[i]
				arg2 = y + h * increment + h * self.RKMatrix[i,i] * x
				return self.function(arg1, arg2)  - x
				
			increments[:,i] = fsolve(func, y, increment)
		return increments
