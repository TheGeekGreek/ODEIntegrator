# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from numpy.linalg import norm
from numpy import finfo, reshape, sqrt, zeros_like, tile, dot
from warnings import warn

from ExplicitRungeKutta import *

class EmbeddedRungeKutta(ExplicitRungeKutta):
    def __init__(
            self,  
            rk_matrix, 
            rk_weights, 
            function, 
            initial_value, 
            settings
        ):
        """Initializer of an instance of the class EmbeddedRungeKutta"""
        RungeKutta.__init__(
                self, 
                rk_matrix,
                rk_weights[0],
                function,
                initial_value,
                settings
            )

        self.rk_weights_hat = rk_weights[1]
                
        return None

    def _calculate_initial_step(self, tstart, abstol, reltol, params):
        """
        Calculates an optimal starting step size.

        Estimates a good choice of a starting stepsize for an embedded Runge-
        Kutta method based on [Hai00]_, page 169.

        References:
        ----------
        .. [Hai00]  E. Hairer, S.P. Norsett and G Wanner. Solving ordinary 
                    differential equations I: Nonstiff problems. Springer. 
                    Second revised edition. 2000.

        """
        n = self.initial_value.size
        power = self.rk_matrix.shape[0] - 1
        evaluation1 = self.function(tstart, self.initial_value, *params)
         
        scaling = array(abstol)
            
        for i in range(len(evaluation1)):
            scaling[i] += abs(self.initial_value[i]) * reltol[i]

        d0 = sqrt(1./n * sum((self.initial_value/scaling)**2))
        d1 = sqrt(1./n * sum((evaluation1/scaling)**2))

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 1e-2 * (d0/d1)

        y1 = self.initial_value + h0 * evaluation1
        evaluation2 = self.function(self.initial_value + h0, y1, *params)
        d2 = sqrt(1./n * sum(((evaluation2 - evaluation1)/scaling)**2))/h0
            
        if max(d1,d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (1e-2/max(d1,d2))**(1./power)

        return min(1e+2 * h0, h1)

    def integrate(self, tstart, tend):
        """
        Integrating method for an embedded RK method.
                
        This implementation follows page 167 & 168 in [Hai00]_.

        Parameters:
        ----------
        Foor the parameters xStart, xEnd, number_of_steps and verbose have
        a look at the corresponding method of the class RungeKutta.

        References:
        ----------
        .. [Hai00]  E. Hairer, S.P. Norsett and G Wanner. Solving ordinary 
                    differential equations I: Nonstiff problems. Springer. 
                    Second revised edition. 2000.
        """
        h = self._calculate_initial_step(
                    tstart, 
                    self.settings['abstol'], 
                    self.settings['reltol'],
                    self.settings['params']
                )

        #Constants
        n = self.initial_value.size
        nu = self.rk_matrix.shape[0]
        power = nu - 1
        y0 = self.initial_value
        fac = (.25)**(1./power)
        facmax = 2.
        t = tstart

        #Outputsavings
        y = [y0]
        time = [t]
        time_increments = []
        rejected = 0
        successfull = False

        #Main loop
        while t < tend:
            if t + h >= tend:
                h = tend - t
                successfull = True
                 
            elif h > self.settings['h_max'] or h < self.settings['h_min']:
                message = 'Solving might not be successful. Try to set the' \
                        ' setting for maximal and minimal stepsize manually' \
                        ' by using **kwargs in integration.'
                warn(message, Warning)

            elif len(time_increments) > self.settings['max_steps']:
                message = 'Solving has not been successful. The iterative' \
                        ' integration loop exited at time t = %s before the' \
                        ' endpoint tEnd = %s was reached. Try to set the' \
                        ' setting for maximal steps manually by using' \
                        ' **kwargs in integration. Changing the maximal' \
                        ' stepsize can drastically increase the runtime' \
                        ' behaviour of the algorithm.'%(t,tend)
                print message
                break
            
            increments = self._compute_increments(y0, t, h, *self.settings['params']) 
            y1 = y0 + h * dot(increments, self.rk_weights)
            y_hat1 = y0 + h * dot(increments, self.rk_weights_hat)
                        
            scaling = array(self.settings['abstol'])
                        
            for i in range(n):
                factor = max(abs(y0[i]), abs(y1[i])) 
                scaling[i] += factor * self.settings['reltol'][i]

            error = sqrt(1./n * sum(((y1 - y_hat1)/scaling)**2))
            
            if error >= finfo(float).tiny:
                r = min(facmax, max(.1,fac * (1./error)**(1./power)))
            else:
                r = facmax

            h_new = h * r                   
                        
            if error <= 1.:
                time_increments.append(h)
                
                #Local extrapolation
                y0 = y_hat1
                
                t += h
                h = h_new
                facmax = 2.
                y.append(y0)
                time.append(t)
                
                if self.settings['verbose']:
                    self._progress(t - tstart, tend - tstart)

            else:
                rejected += 1
                h = h_new
                facmax = 1.
       
        if self.settings['verbose']:
            if successfull:
                print 'Stepsize control algorithm was successfull.'
            else:
                print 'Stepsize control algorithm was not successfull.'
            print 'The largest possible step was: %s'%self.settings['h_max']
            print 'The smallest possible step was: %s'%self.settings['h_min']
            print 'Number of steps: %d'%(len(time_increments))
            print 'Number of function calls: %d'%(6 * (len(time) - 1))
            print 'Maximal stepsize achieved in integration: %s'%max(time_increments)
            print 'Minimal stepsize achieved in integration: %s'%min(time_increments)
            print 'Number of rejected steps: %d'%rejected
                
        return array(time_increments), array(time), array(y).T
