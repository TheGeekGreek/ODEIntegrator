# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from numpy.linalg import norm
from numpy import reshape, sqrt, zeros_like, tile

from ExplicitRungeKutta import *

class EmbeddedRungeKutta(ExplicitRungeKutta):
    def __init__(self,  RKMatrix, RKWeights, function, initialValues):
        """Initializer of an instance of the class EmbeddedRungeKutta"""
        RungeKutta.__init__(self, RKMatrix, RKWeights[0], function, initialValues)
        self.RKWeightsHat = RKWeights[1]
                
        return None

    def _calculate_initial_step(self, tStart, abstol, reltol):
        """
        Calculates an optimal starting step size.

        Estimates a good choice of a starting stepsize for an embedded Runge-
        Kutta method. It is basesd on [Hai00]_, page 169.

        References:
        ----------
        .. [Hai00]  E. Hairer, S.P. Norsett and G Wanner. Solving ordinary 
                    differential equations I: Nonstiff problems. Springer. 
                    Second revised edition. 2000.

        """
        n = self.initialValues.size
        power = self.RKMatrix.shape[0] - 1
        evaluation1 = self.function(tStart, self.initialValues)
         
        scaling = zeros_like(evaluation1)
            
        for i in range(len(evaluation1)):
            scaling[i] = abstol[i] + abs(self.initialValues[i]) * reltol[i]

        d0 = sqrt(1./n * sum((self.initialValues/scaling)**2))
        d1 = sqrt(1./n * sum((evaluation1/scaling)**2))

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 1e-2 * (d0/d1)

        y1 = self.initialValues + h0 * evaluation1
        evaluation2 = self.function(self.initialValues + h0, y1)
        d2 = sqrt(1./n * sum(((evaluation2 - evaluation1)/scaling)**2))/h0
            
        if max(d1,d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (1e-2/max(d1,d2))**(1./power)

        return min(1e+2 * h0, h1)

    def integrate(self, tStart, tEnd, number_of_steps, verbose, **kwargs):
        """
        Integrating method for an embedded RK method.
                
        This implementation follows page 167 & 168 in [Hai00]_. 

        References:
        ----------
        .. [Hai00]  E. Hairer, S.P. Norsett and G Wanner. Solving ordinary 
                    differential equations I: Nonstiff problems. Springer. 
                    Second revised edition. 2000.


        """
        #self._has_func_initial()

        settings = {
            'abstol':tile(array([1e-5]), self.initialValues.size),
            'reltol':tile(array([1e-5]), self.initialValues.size),
            'h_min':1e-15,
            'h_max':1e+2,
            'max_steps':int(1e+4)
        }

        #Handling the **kwargs provided by the user     
        if kwargs:
            for key in kwargs:
                if key in settings:
                    settings[key] = kwargs[key]
            else:
                raise ValueError, 'Not supported keyword-argument' \
                                    ' %s provided.'%(key)
        
        h = self._calculate_initial_step(
                tStart, 
                settings['abstol'], 
                settings['reltol']
                )

        #Constants
        n = self.initialValues.size
        nu = self.RKMatrix.shape[0]
        power = nu - 1
        y0 = self.initialValues
        fac = (.25)**(1./power)
        facmax = 2.
        t = tStart
        y = []
        time = []
        timeIncrements = []
        rejected = 0
        #Main loop
        while t < tEnd:
            if t + h >= tEnd:
                h = tEnd - t 
                       
            increments = self._compute_increments(y0, t, h) 
            y1 = y0 + h * numpy.sum(increments * self.RKWeights, axis = 1)
            yHat1 = y0 + h * numpy.sum(increments * self.RKWeightsHat, axis = 1)
                        
            scaling = zeros_like(settings['abstol'])
                        
            for i in range(n):
                    factor = max(abs(y0[i]), abs(y1[i]))
                    arg1 = settings['abstol'][i] 
                    arg2 = factor * settings['reltol'][i]
                    scaling[i] = arg1 + arg2

            error = sqrt(1./n * sum(((y1 - yHat1)/scaling)**2))
            r = min(facmax, max(.1,fac * (1./error)**(1./power)))
            h_new = h * r                   
                        
            if error <= 1.:
                timeIncrements.append(h)
                y0 = yHat1
                t += h
                h = h_new
                facmax = 2.
                time.append(t)
                y.append(y0)

            else:
                rejected += 1
                h = h_new
                facmax = 1.
                
        if verbose:
            print 'The largest possible step was: %.2f'%settings['h_max']
            print 'The smallest possible step was: %.9f'%settings['h_min']
            print 'Number of steps: %d'%(len(timeIncrements))
            print 'Number of function calls: %d'%(6 * (len(time) - 1))
            print 'Maximal stepsize achieved in integration: %.8f'%max(timeIncrements)
            print 'Minimal stepsize achieved in integration: %.8f'%min(timeIncrements)
            print 'Number of rejected steps: %s'%rejected
                
        return array(timeIncrements), array(time), array(y).T

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
