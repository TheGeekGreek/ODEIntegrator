# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
import numpy
from numpy import shape, sign, zeros, linspace, hstack, array

class RungeKutta(object):
	def __init__(self, RKMatrix, RKWeights, function, initialValues):
		"""
		Initializer of an instance of the class RungeKutta.

		Notes:
		-----
		The naming conventions follow [Ise96]_, page 38.

		References:
		----------
		..  [Ise96] Arieh Iserles. A first course in the numerical analysis of
					differential equations. Cambridge texts in applied mathe-
					matics, 1996.

		"""
		self.RKMatrix = RKMatrix
		self.RKWeights = RKWeights
		self.function = function
		self.initialValues = initialValues
		self.calculate_RK_nodes()
		
		return None
	
	def calculate_RK_nodes(self):
		nu = self.RKMatrix.shape[0]
		RKNodes = zeros(nu)
		for j in xrange(nu):
			RKNodes[j] = sum(self.RKMatrix[j,0:nu])
		self.RKNodes = RKNodes
		
		return None

	def integrate(self, tStart, tEnd, steps, verbose, **kwargs):
		if kwargs:
			raise ValueError, 'No keyword-arguments allowed.'

		[t, h] = linspace(tStart, tEnd, steps + 1, retstep = True)
		y = zeros((self.initialValues.size, t.size))
		y[:, 0] = self.initialValues

		for k in xrange(steps):
			l = self._compute_increments(y[:,k], t[k], h)
			y[:, k + 1] = y[:,k] + h * numpy.sum(l * self.RKWeights, axis = 1) 
		return h, t, y	
		

	def __str__(self):
		"""
		Method for printing an instance of the class RungeKutta.

		Prints the butcher tableau in an fashionable way. This means that

									c|A
									---
									 |b

		is printed where A corresponds to the attribute RKMatrix, c to 
		RKNodes and b to RKWeights.
		"""
		string = ''
		for i in xrange(len(self.RKNodes)):
			string += '+%.3f'%self.RKNodes[i] + ' | '
			for element2 in self.RKMatrix[i]:
				if element2 >= 0:
					string += '+%.3f '%element2
				else:
					string += '%.3f '%element2
			string += '\n'
		length = len(string.split('\n')[0])
		string += length * '-' + '\n'
		string += '       | '
		for element in self.RKWeights:
			if element >= 0:
				string += '+%.3f '%element
			else:
				string += '%.3f '%element
		string += '\n'
		return string
