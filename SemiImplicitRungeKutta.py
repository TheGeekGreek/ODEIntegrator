from scipy.optimize import fsolve

from RungeKutta import *

class SemiImplicitRungeKutta(RungeKutta):
	def build():
		return None

	def _compute_increments(self, y, t, h):
		s = self.A.shape[0]
		l = zeros((self.initialValues.size,s)) 
		for i in xrange(s):
			l[:,i] = self.func(t + h * self._c[i], y + h * np.sum( self._A[i,0:self._s] * l[:,0:self._s], axis = 1))
			
			F = lambda x: self.func(t + h * self._c[i], y + h * np.sum( self._A[i,0:i] * l[:,0:i], axis = 1) + h * self._A[i,i] * x) - x 
			l[:,i] = scipy.optimize.fsolve(F,y)
		return l
