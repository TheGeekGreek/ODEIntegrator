from scipy.optimize import fsolve
from numpy import tile

from RungeKutta import *

class ImplicitRungeKutta(RungeKutta):
	def build():
		return None

	def _compute_increments(self, y, t, h):
		nu = self.A.shape[0]
		n = self.initialValues.size
		increments = zeros((n * nu))
		
		def F(x):
			for i in xrange(nu):
				increments = zeros((n,))
				for j in xrange(nu):
					increments += self.A[i,j] * x[j * n:(j + 1) * n]	
				increments[i * n:(i + 1) * n] = self.function(t + h * self.c[i], y + h * increments) - x[i * n:(i + 1) * n]			
			return increments
			
		init = tile(y, nu)
		increments = fsolve(F, init.T)
		increments = increments.reshape((n, nu)).T

		return increments
