# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from numpy.linalg import norm
from numpy import reshape, mean, sqrt, zeros_like, tile

from ExplicitRungeKutta import *

class EmbeddedRungeKutta(ExplicitRungeKutta):
	def __init__(self,  RKMatrix, RKWeights, function, initialValues):
		"""Initializer of an instance of the class EmbeddedRungeKutta"""
		RungeKutta.__init__(self, RKMatrix, RKWeights[0], function, initialValues)
		self.RKWeights_tilde = RKWeights[1]
		self.settings = {
			'abstol':tile(array([1e-5]), self.initialValues.size),
			'reltol':tile(array([1e-5]), self.initialValues.size),
			'h_min':1e-20,
			'h_max':1e+1,
			'max_steps':int(1e+4)
		}
		
		return None

	def integrate(self, tStart, tEnd, number_of_steps, verbose, **kwargs):
		"""Integrating method for an embedded RK method."""
		#Handling the **kwargs provided by the user	
		if kwargs:
			for key in kwargs:
				if key in self.settings:
					self.settings[key] = kwargs[key]
				else:
					raise ValueError, 'Not supported keyword-argument' \
							' %s provided.'%(key)

		#Calculating the initial time step from number_of_steps
		h = (tEnd - tStart)/float(number_of_steps)
		
		def _norm(x, z):
			n = len(x)
			return sqrt(1./n * sum((x/z)**2))
		
		approx = self.function(tStart, self.initialValues)
		sc = zeros_like(approx)
		for i in range(len(approx)):
			sc[i] = self.settings['abstol'][i] + abs(self.initialValues[i]) * self.settings['reltol'][i]
		d0 = _norm(self.initialValues, sc)
		d1 = _norm(approx, sc)
		if d0 < 1e-5 or d1 < 1e-5:
			h0 = 1e-6
		else:
			h0 = 1e-2 * d0/d1
		y1 = self.initialValues + h0 * approx
		approx2 = self.function(tStart + h0, y1)
		d2 = _norm(approx2 - approx, sc)/h0
		if max(d1,d2) <= 1e-15:
			h1 = max(1e-6, h0 * 1e-3)
		else:
			h1 = (1e-2/max(d1,d2))**(1./5)

		h = min(1e+2 * h0, h1)
			
		#Preventing the initial time-step from being too small
		#if h <= self.settings['h_min']:
		#	h = self.settings['h_min']
                

		#Constants
		n = self.initialValues.size
		nu = self.RKMatrix.shape[0]
		y = self.initialValues.reshape((n,1))
		yTilde = self.initialValues.reshape((n,1))
		fac = (.25)**(1./5)
		#fac = .8
		facmax = 2.
		#Initializing time-variable
		t = tStart
		
		errors = array([])
		timeIncrements = array([])
		timeSpan = array([t])
		
		#Main loop
		while t < tEnd:
                        if timeSpan[-1] + h >= tEnd:
                            h = tEnd - timeSpan[-1]
                        elif h > self.settings['h_max'] or h < self.settings['h_min']:
			    break
                        else:
                            pass
			
                        l = self._compute_increments(yTilde[:,-1], t, h)
			order_pp = yTilde[:,-1] + h * numpy.sum(l * self.RKWeights_tilde, axis = 1)
			yTilde = hstack((yTilde[:,-1].reshape((n,1)), order_pp.reshape((n,1))))
			order_p = y[:,-1] + h * numpy.sum(l[:,:-1] * self.RKWeights[:-1], axis = 1)
			y = hstack((y, order_p.reshape((n,1))))
			
			sc = zeros_like(self.settings['abstol'])

			for i in range(n):
				factor = max(abs(y[i,-2]), abs(y[i,-1]))
				sc = self.settings['abstol'][i] + factor * self.settings['reltol'][i]
			error = _norm(y[:,-1] - yTilde[:,-1], sc)	
			r = min(facmax, max(.5, fac * (1./error)**(1./5)))
			h_new = h * r			
    
			if error <= 1.:
				t += h
				timeSpan = hstack((timeSpan, array(t)))
				h = h_new
				facmax = 2.

			else:
				yTilde = yTilde[:,:-1]
				y = y[:,:-1]
				h = h_new
				facmax = 1.

		if verbose:
			#print 'The upper error bound was: %.10f'%settings['eps_max']
			#print 'The lower error bound was: %.10f'%settings['eps_min']
			print 'The largest possible step was: %.2f'%self.settings['h_max']
			print 'The smallest possible step was: %.9f'%self.settings['h_min']
			print 'Number of steps: %d'%(len(timeSpan) - 1)
			print 'Number of function calls: %d'%(6 * (len(timeSpan) - 1))
			#print 'Maximal stepsize achieved in integration: %.8f'%max(timeIncrements)
			#print 'Minimal stepsize achieved in integration: %.8f'%min(timeIncrements)
			#print 'Average error: %.11f'%mean(errors)
		
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
