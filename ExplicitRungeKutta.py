# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from RungeKutta import *

class ExplicitRungeKutta(RungeKutta):
	def _compute_increments(self, y, t, h, *args):
		"""Computes the increments of one RK step."""
		nu = self.rk_matrix.shape[0]
		increments = zeros((self.initial_value.size, nu))		
		
		for i in xrange(nu):
			increment = dot(increments[:,0:i], self.rk_matrix[i,0:i])
			increments[:,i] = self.function(
									t + h * self.rk_nodes[i], 
									y + h * increment,
									*args
								)
		
		return increments
