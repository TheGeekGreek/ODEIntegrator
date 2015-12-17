# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from RungeKutta import *

class ExplicitRungeKutta(RungeKutta):
	def _compute_increments(self, y, t, h):
		nu = self.RKMatrix.shape[0]
		increments = zeros((self.initialValues.size, nu))		
		
		for i in xrange(nu):
			increment = numpy.sum(self.RKMatrix[i,0:i] * increments[:,0:i], axis = 1)
			increments[:,i] = self.function(t + h * self.RKNodes[i], y + h * increment)
		
		return increments
