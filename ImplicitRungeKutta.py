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
	def _compute_increments(self, y, t, h, *args):
		nu = self.rk_matrix.shape[0]
		n = self.initial_value.size

		def func(x):
			stacked = zeros((n * nu))
			for i in range(nu):
				pre_i = dot(x.reshape((nu, n)).T[:,0:i], self.rk_matrix[i,0:i])
				post_i = dot(x.reshape((nu, n)).T[:,i+1:nu], self.rk_matrix[i,i+1:nu])
				arg1 = t + h * self.rk_nodes[i]
				arg2 = y + h * (pre_i + post_i + self.rk_matrix[i,i] * x[n * i:n * (i + 1)])
				stacked[n * i:n * (i + 1)] = self.function(arg1, arg2, *args) - x[n * i:n * (i + 1)]
			return stacked

		return fsolve(func, tile(y, nu)).reshape((nu, n)).T
