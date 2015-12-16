# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from numpy.linalg import norm
from numpy import reshape, mean

from ExplicitRungeKutta import *

class EmbeddedRungeKutta(ExplicitRungeKutta):
	def __init__(self,  RKMatrix, RKWeights, function, initialValues):
		"""Initializer of an instance of the class EmbeddedRungeKutta"""
		RungeKutta.__init__(self, RKMatrix, RKWeights[0], function, initialValues)
		self.RKWeights_tilde = RKWeights[1]
		
		return None

	def integrate(self, tStart, tEnd, number_of_steps, verbose, **kwargs):
		"""
		Integrating method for an embedded RK method.

		Parameters:
		----------
		For the parameters tStart, tEnd, number_of_steps and verbose have a
		look at the corresponding docstring of the method integrate in the class 
		RungeKutta. Worth to metion are the possible keyword-arguments:

		eps_min : float
				Minimal error bound.

		eps_max : float
				Maximal error bound.

		h_min : float
				Minimal stepsize of integration.

		h_max : float
				Maximal stepsize of integration.

		max_steps : int
				Maximal possible number of steps achieved in integration.

		Returns:
		-------
		Have a look at the docstring of the method integrate of the class 
		RungeKutta.
		"""
		settings = {
			'eps_min':1e-10,
			'eps_max':1e-9,
			'h_min':1e-10,
			'h_max':1e+2,
			'max_steps':int(1e+4)
		}
		
		if kwargs:
			for key in kwargs:
				if key in settings:
					settings[key] = kwargs[key]
				else:
					raise ValueError, 'Not supported keyword-argument' \
							' %s provided.'%(key)
		
		#Calculating the initial time step
		h0 = (tEnd - tStart)/float(number_of_steps)
		
		if h0 <= settings['h_min']:
			h0 = settings['h_min']
		
		n = self.initialValues.size
		y = self.initialValues.reshape((n,1))
		z = self.initialValues.reshape((n,1))
		t = tStart
		
		errors = array([])
		timeIncrements = array([])
		timeSpan = array([t])

		#Main loop
		while t < tEnd:
			timeSpan = hstack((timeSpan, array(t)))
			l = self._compute_increments(y[:,-1], t, h0)
			order_pp = (y[:,-1] + h0 * numpy.sum(l * self.RKWeights_tilde, axis = 1))
			y = hstack((y, order_pp.reshape((n,1))))
			order_p = (z[:,-1] + h0 * numpy.sum(l[:,:-1] * self.RKWeights[:-1], axis = 1))
			z = hstack((z[:,-1].reshape((n,1)), order_p.reshape((n,1))))
			
			#Calculating the truncation error in euclidean standard norm
			epsilon = norm(y[:,-1] - z[:,-1], 2)
			
			if epsilon < settings['eps_min']:
				errors = hstack((errors, array(epsilon)))
				h_new = 2 * h0
				if h_new < settings['h_max']:
					h0 = h_new
				else:
					h0 = settings['h_max']
				t += h0				
				timeIncrements = hstack((timeIncrements, h0))

			elif epsilon > settings['eps_max']:
				h_new = h0/2.
				if h_new > settings['h_min']:
					h0 = h_new
					z = z[:,:-1]
					y = y[:,:-1]
					timeSpan = timeSpan[:-1]
				else:
					h0 = settings['h_min']
					t += h0
					timeIncrements = hstack((timeIncrements, h0))

			else:
				errors = hstack((errors, array(epsilon)))
				h0 *= 1.1			
				t += h0
				timeIncrements = hstack((timeIncrements, h0))

			if len(timeIncrements) >= settings['max_steps']:
				print 'Maximum number of steps reached. This method is' \
						' possibly not appropriate for this integration problem.'
				print 'Method failed.'
				break

		#Calculating the last step
		dt = tEnd - timeSpan[-1]
		l = self._compute_increments(y[:,-1], timeSpan[-1], dt)
		order_pp = y[:,-1] + dt * numpy.sum(l * self.RKWeights_tilde, axis = 1)
		y = hstack((y, order_pp.reshape((n,1))))
		timeSpan = hstack((timeSpan, array([tEnd])))

		if verbose:
			print 'The upper error bound was: %.10f'%settings['eps_max']
			print 'The lower error bound was: %.10f'%settings['eps_min']
			print 'The largest possible step was: %.2f'%settings['h_max']
			print 'The smallest possible step was: %.9f'%settings['h_min']
			print 'Number of steps: %d'%(len(timeSpan) - 1)
			print 'Number of function calls: %d'%(6 * (len(timeSpan) - 1))
			print 'Maximal stepsize achieved in integration: %.8f'%max(timeIncrements)
			print 'Minimal stepsize achieved in integration: %.8f'%min(timeIncrements)
			print 'Average error: %.11f'%mean(errors)
		
		return timeIncrements, timeSpan, y

	def __str__(self):
		string = RungeKutta.__str__(self)
		string += '       | '
		for element in self.RKWeights_tilde:
			if element >= 0:
				string += '+%.3f '%element
			else:
				string += '%.3f '%element
		string += '\n'
		return string
