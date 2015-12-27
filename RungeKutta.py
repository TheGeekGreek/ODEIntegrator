# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from sys import stdout
from numpy import dot, shape, sign, zeros, linspace, hstack, array

class RungeKutta(object):
	def __init__(self, rk_matrix, rk_weights, function, initial_value):
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
		self.rk_matrix = rk_matrix
		self.rk_weights = rk_weights
		self.function = function
		self.initial_value = initial_value
		self._calculate_rk_nodes()
		
		return None
	
	def _calculate_rk_nodes(self):
		nu = self.rk_matrix.shape[0]
		rk_nodes = zeros(nu)
		for j in xrange(nu):
			rk_nodes[j] = sum(self.rk_matrix[j,0:nu])
		self.rk_nodes = rk_nodes
		
		return None
		
	def integrate(self, tstart, tend, **settings):
		t, h = linspace(tstart, tend, settings['steps'] + 1, retstep = True)
		y = zeros((self.initial_value.size, t.size))
		y[:, 0] = self.initial_value

		for k in xrange(settings['steps']):
			increments = self._compute_increments(y[:,k], t[k], h, *settings['params'])
			y[:, k + 1] = y[:,k] + h * dot(increments, self.rk_weights)

			if settings['verbose']:
				self._progress(t[k + 1] - tstart, tend - tstart)

		if settings['verbose']:
			print 'RK method was successfull.'
		
		return h, t, y	
	
	def _progress(self, count, total):
		"""
		Prints the progress of integration on console.

		An ASCII-progress bar on console. This is slightly modified code 
		taken from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3. 
		With a big thank to Vladimir Ignatyev who kindly allowed me to use his
		code.
		"""
		bar_len = 30
		filled_len = int(round(bar_len * count / float(total)))

		percents = round(100.0 * count / float(total), 3)
		bar = '=' * filled_len + '-' * (bar_len - filled_len)

		stdout.write('\r[%s] %s%s ' % (bar, percents, '%'))
		stdout.flush()

		return None
