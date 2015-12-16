# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from RungeKutta import *

class ExplicitRungeKutta(RungeKutta):
	def build(self):
		return None

	def _compute_increments(self, y, t, h):
		nu = self.RKMatrix.shape[0]
		increments = zeros((self.initialValues.size, nu))		
		for i in xrange(nu):
			increment = y + h * numpy.sum(self.RKMatrix[i,0:nu] * increments[:,0:nu], axis = 1)
			increments[:,i] = self.function(t + h * self.RKNodes[i], increment)
		return increments

		"""
		def increment(self, y, h, l, i):
		return y + h * numpy.sum(self.A[i,0:self.A.shape[0]] * l[:,0:self.A.shape[0]], axis = 1)


		def _compute_increments(self, y, t, h):
		l = zeros((self.initialValues.size,self.A.shape[0]))
		for i in xrange(self.A.shape[0]):
			l[:,i] = self.function(t + h * self.c[i], self.increment(y, h, l, i))
		return l
		"""
