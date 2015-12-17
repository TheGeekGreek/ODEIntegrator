# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from scipy.optimize import fsolve
from numpy import tile

from RungeKutta import *

class ImplicitRungeKutta(RungeKutta):
	def _compute_increments(self, y, t, h):
		nu = self.RKMatrix.shape[0]
		n = self.initialValues.size

		def func(x):
			stacked = zeros((n * nu))
			for i in range(nu):
				pre_i = numpy.sum(self.RKMatrix[i,0:i] * x.reshape((nu, n)).T[:,0:i], axis = 1)
				post_i = numpy.sum(self.RKMatrix[i,i+1:nu] * x.reshape((nu, n)).T[:,i+1:nu], axis = 1)
				arg1 = t + h * self.RKNodes[i]
				arg2 = y + h * (pre_i + post_i + self.RKMatrix[i,i] * x[n * i:n * (i + 1)])
				stacked[n * i:n * (i + 1)] = self.function(arg1, arg2) - x[n * i:n * (i + 1)]
			return stacked

		return fsolve(func, tile(y, nu)).reshape((nu, n)).T
