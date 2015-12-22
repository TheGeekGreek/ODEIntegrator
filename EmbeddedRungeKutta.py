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
                        'abstol':tile(array([1e-4]), self.initialValues.size),
                        'reltol':tile(array([1e-4]), self.initialValues.size),
                        'h_min':1e-40,
                        'h_max':1e+2,
                        'max_steps':int(1e+4)
                    }
                
            return None

        def _calculate_initial_step(self, tStart):
            """
            Calculates an optimal starting step size.

            Estimates a good choice of a starting stepsize for an adaptive Runge-
            Kutta method. It is basesd on [Hai00]_, page 169.

            References:
            ----------
            .. [Hai00]  E. Hairer, S.P. Norsett and G Wanner. Solving ordinary 
                        differential equations I: Nonstiff problems. Springer. 
                        Second revised edition. 2000.

            """
            n = float(self.initialValues.size)
            evaluation1 = self.function(tStart, self.initialValues)
            
            sc = zeros_like(evaluation1)
            
            for i in range(len(evaluation1)):
                arg1 = self.settings['abstol'][i] 
                arg2 = abs(self.initialValues[i]) * self.settings['reltol'][i]
                sc[i] = arg1 + arg2

            d0 = sqrt(1./n * sum((self.initialValues/sc)**2))
            d1 = sqrt(1./n * sum((evaluation1/sc)**2))

            if d0 < 1e-5 or d1 < 1e-5:
                h0 = 1e-6
            else:
                h0 = 1e-2 * (d0/d1)

            y1 = self.initialValues + h0 * evaluation1
            evaluation2 = self.function(self.initialValues + h0, y1)
            d2 = sqrt(1./n * sum(((evaluation2 - evaluation1)/sc)**2))/h0
            
            if max(d1,d2) <= 1e-15:
                h1 = max(1e-6, h0 * 1e-3)
            else:
                power = self.RKMatrix.shape[0] - 1
                h1 = (1e-2/max(d1,d2))**(1./power)

            return min(1e+2 * h0, h1)

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
                h = self._calculate_initial_step(tStart)
                
                print h
                def _norm(x, z):
                        n = len(x)
                        return sqrt(1./n * sum((x/z)**2))

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
                        yTilde = hstack((yTilde, order_pp.reshape((n,1))))
                        order_p = y[:,-1] + h * numpy.sum(l[:,:-1] * self.RKWeights[:-1], axis = 1)
                        y = hstack((y, order_p.reshape((n,1))))
                        
                        sc = zeros_like(self.settings['abstol'])
                        for i in range(n):
                                factor = max(abs(y[i,-2]), abs(y[i,-1]))
                                arg1 = self.settings['abstol'][i] 
                                arg2 = factor * self.settings['reltol'][i]
                                sc = arg1 + arg2

                        error = _norm(y[:,-1] - yTilde[:,-1], sc)
                        print error
                        r = min(facmax, max(.1,fac * (1./error)**(1./5)))
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
