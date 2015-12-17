"""
ODEIntegrator - A class for integrating systems of first order explicite 
ordinary differential equations using various Runge Kutta methods. 
Implemented are explicite, semi-implicite, implicite and embedded
methods up to the order five.

Author:
-------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from numpy import array, ndarray, sqrt
from time import time
from types import FunctionType

#Wildcard import from own modules
from ExplicitRungeKutta import *
from SemiImplicitRungeKutta import *
from ImplicitRungeKutta import *
from EmbeddedRungeKutta import *

__all__ = ['ODEIntegrator']
__version__ = '0.1'
__docformat__ = 'restructuredtext en'

######################################################################
#User interface														 #
######################################################################

class ODEIntegrator(object):
	"""
	An interface for integrating systems of first order ODEs.

	Solve a system of first order ODEs of the form y'(x) = f(x,y(x))
	where f is in C(U;R^n) and U is a subset of R^{n+1} with n in 
	N_{>0} and a given Cauchy-problem of the form y(xStart) = 
	y_0 in R^n on the interval [xStart, xEnd] in R.

	Attributes:
	----------
	function : function
			The function f given in y'(x) = f(x,y(x)).
	
	initialValues: numpy.ndarray
			The initial values provided by the Cauchy-problem. 

	Example:
	-------
	>>> from ODEIntegrator import *
	>>> from numpy import zeros_like, sin, array, pi
	>>> def right_hand_side(t, y):
	...     values = zeros_like(y)
	...     values[0] = y[1]
	...     values[1] = -sin(y[0])
	...     return values
	... 
	>>> initialValues = array([-pi/3,0])
	>>> tStart = 0.
	>>> tEnd = 10.
	>>> number_of_steps = 1e+3
	>>> instance = ODEIntegrator(right_hand_side, initialValues)
	>>> instance.build('EERK45')
	>>> h, t, y = instance.integrate(
	...     tStart,
	...     tEnd,
	...     number_of_steps,
	...     return_steps = True
	... )
	>>> print h
	[ 0.02        0.04        0.08        0.16        0.176       0.1936
	  0.21296     0.234256    0.2576816   0.28344976  0.31179474  0.34297421
	  0.37727163  0.41499879  0.45649867  0.50214854  0.55236339  0.60759973
	  0.66835971  0.73519568  0.80871525  0.88958677  0.97854545  0.5382
	  0.59202     0.651222  ]
	>>> print t
	[  0.           0.02         0.06         0.14         0.3          0.46
   	   0.636        0.8296       1.04256      1.276816     1.5344976
   	   1.81794736   2.1297421    2.47271631   2.84998794   3.26498673
   	   3.7214854    4.22363394   4.77599734   5.38359707   6.05195678
   	   6.78715246   7.5958677    8.48545447   8.9747272    9.51292719
	  10.10494719  -0.10494719]
	>>> print y
	[[-1.04719755 -1.04715425 -1.04680785 -1.04507622 -1.03746393 -1.00575377
	  -0.94595912 -0.85128141 -0.71479064 -0.53069925 -0.29673679 -0.0178017
	   0.28979344  0.59553947  0.85581901  1.01878781  1.02991575  0.83839432
	   0.4181342  -0.17544336 -0.75073425 -1.04240834 -0.83802009 -0.09305638
	   0.38613243  0.8092958   1.03680803  1.01761073]
	 [ 0.          0.00866018  0.02597881  0.060597    0.12965915  0.26627716
	   0.4122263   0.56395023  0.71446558  0.85141472  0.95529712  0.99984319
	   0.95739869  0.80975267  0.55785857  0.2209144  -0.17261211 -0.5808086
	  -0.90980891 -0.9846423  -0.68004467 -0.0914493   0.58155996  0.99620283
	   0.92404069  0.61732608  0.13790598  0.22776722]]
	"""
	def __init__(self, function, initialValues = array([])):
		"""Initializer of an instance of the class ODEIntegrator."""
		if isinstance(function, FunctionType):
			self.function = function
		else:
			raise ValueError, 'Expected an object of type function' \
					' got %s instead.'%type(function)
		
		if isinstance(initialValues, ndarray):
			self.initialValues = initialValues
		else:
			raise ValueError, 'Expected an object of type numpy ' \
					' ndarray got %s instead.'%type(initial)
		return None	

	def _get_RK_tableaux(self, method):
		"""
		Returns the RK tableaux of a certain method.	

		Parameters:
		----------
		method : string
				Name of the method.
	
		Returns:
		-------
		(A , b) : tuple
				Tuple of the RK weights and RK matrix where each is
				a numpy.ndarray. In the addaptive case b = (b_1, b_2) where 
				b_1 are the RK weights of the lower order method and b_2 are
				the weights of the higher order method.

		Notes:
		-----
		====== ===== ============================= ==================
        Method Order Full name             		   Reference
        ====== ===== ============================= ==================
        FE     1     Forward Euler		           [Ise96]_, page 4 
        ET     2     Explicit trapezoidal or Heun  [Ise96]_, page 39
        ERK2   2     Ralston      				   [Ise96]_, page 39
        EM     2     Explicit midpoint 			   [Ise96]_, page 39
		ERK3   3     Classical RK order 3		   [Ise96]_, page 40
		NYS    3	 Nystrom 					   [Ise96]_, page 40
		SDIRK  3     Singly diagonal implicit	   [Hai00]_, page 207
        ERK4   4     "The" Runge-Kutta method	   [Hai00]_, page 138
        ERK3/8 4     3/8-rule    				   [Hai00]_, page 138 
        BE     1     Backward Euler     		   [Hai00]_, page 205
        IM     2     Implicit Midpoint rule 	   [Hai00]_, page 205 
        IT     2     Implicit trapezoidal    	   
        GL4    4     Gauss-Legendre    			   [Ise96]_, page 47
        GL6    6     Gauss-Legendre  			   [Ise96]_, page 47
		KB8    8     Kuntzmann & Butcher           [Hai00]_, page 209
		EERK12 2(1)  Heun-Euler
        EERK23 3(2)  Simple embedded RK pair	   [Ise96]_, page 84
		EERK45 5(4)  Fehlberg                      [Ise96]_, page 84
        ====== ===== ============================= ==================

		References:
		----------
		.. [Ise96] Arieh Iserles. A first course in the numerical analysis of
				   differential equations. Cambridge texts in applied mathe-
				   matics, 1996.

		.. [Hai00] E. Hairer, S.P. Norsett and G Wanner. Solving ordinary 
				   differential equations I: Nonstiff problems. Springer. 
				   Second revised edition. 2000.
		"""
		#Constants of Kuntzmann & Butcher, order 8
		omega1 = 1./8 - sqrt(30)/144 
		omega2 = 1./2 * sqrt((15 + 2 * sqrt(30))/35)
		omega3 = omega2 * (1./6 + sqrt(30)/24)
		omega4 = omega2 * (1./21) + (5 * sqrt(30)/168)
		omega5 = omega2 - 2 * omega3
		omegaPrime1 = 1./8 + sqrt(30)/144 
		omegaPrime2 = 1./2 * sqrt((15 + 2 * sqrt(30))/35)
		omegaPrime3 = omegaPrime2 * (1./6 + sqrt(30)/24)
		omegaPrime4 = omegaPrime2 * (1./21) + (5 * sqrt(30)/168)
		omegaPrime5 = omegaPrime2 - 2 * omegaPrime3

		butcherArray = {
			'FE':(array([[0.]]),array([1.])),
			'ET':(array([[0.,0.],[1.,0.]]), array([1./2,1./2])),
			'ERK2':(array([[0.,0.],[2./3,0.]]), array([1./4, 3./4])),
			'EM':(array([[0.,0.],[1./2,0.]]),array([0.,1.])),
			'ERK3':(
					array([[0,0,0],[1./2,0.,0.],[-1.,2.,0.]]),
					array([1./6,4./6,1./6])
				),
			'NYS':(
					array([[0.,0.,0.],[2./3,0.,0.],[0.,2./3,0.]]),
					array([1./4,3./8,3./8])
				),
			'SDIRK':(
					array([
							[(3-sqrt(3))/6, 0.],
							[1 - (3 - sqrt(3))/3, (3 - sqrt(3))/6]
						]),
					array([1./2,1./2])
				),
			'ERK4':(
				array([
						[0.,0.,0.,0.],
						[1./2,0.,0.,0.],
						[0.,1./2,0.,0.],
						[0., 0., 1., 0.]
					]),
				array([1./6, 1./3, 1./3, 1./6])
				),
			'ERK3/8':(
				array([
						[0,0,0,0],
						[1./3,0,0,0],
						[-1./3,1,0,0],
						[1,-1,1,0]
					]),	
				array([1./8, 3./8, 3./8, 1./8])
				),
			'BE':(array([[1.]]),array([1.])),
			'IM':(array([[1./2]]),array([1.])),
			'IT':(array([[0,0],[1./2,1./2]]),array([1./2,1./2])),
			'GCM4':(
				array([
						[1./4, 1./4 - 1./6 * sqrt(3)], 
						[1./4 + 1./6 * sqrt(3), 1./4]
					]),
				array([1./2,1./2])
				),
			'GCM6':(
				array([
						[ 
							5.0/36., 
							2.0/9 - sqrt(15)/15.,
							5.0/36. - sqrt(15)/30.
						],
              			[ 
							5.0/36.0 + sqrt(15)/24., 
							2.0/9, 5.0/36.0 - sqrt(15)/24.
						],
             			[ 
							5./36. + sqrt(15)/30.,
							2./9 + sqrt(15)/15., 5.0/36.
						]	
					]),
				array([5./18., 4./9., 5./18.]),
				),
			'KB8':(
					array([
							[
								omega1, 
								omegaPrime1 - omega3 + omegaPrime4, 
								omegaPrime1 - omega3 - omegaPrime4,
								omega1 - omega5
							],
							[
								omega1 - omegaPrime3 + omega4,
								omegaPrime1,
								omegaPrime1 - omegaPrime5,
								omega1 - omegaPrime3 - omega4
							],
							[	
								omega1 + omegaPrime3 + omega4,
								omegaPrime1 + omegaPrime5,
								omegaPrime1,
								omega1 + omegaPrime3 - omega4
							],
							[
								omega1 + omega5,
								omegaPrime1 + omega3 + omegaPrime4,
								omegaPrime1 + omega3 - omegaPrime4,
								omega1
							]
						]),
					2 * array([omega1, omegaPrime1, omegaPrime1, omega1])
				),
			'EERK12':(
					array([[0.,0.],[1.,0.]]),
					(array([1.,0.]), array([1./2,1./2]))
				),
			'EERK23':(
					array([[0.,0.,0.],[2./3,0.,0.],[0.,2./3,0.]]),
					(array([1./4,3./4, 0.]),array([1./4,3./8,3./8]))
				),
			'EERK45':(
				array([
					[0., 0., 0., 0., 0., 0.],
					[1./4, 0., 0., 0., 0., 0.],
					[3./32, 9./32, 0., 0., 0., 0.],
					[1932./2197, -7200./2197, 7296./2197, 0., 0., 0.],
					[439./216, -8., 3680./513, -845./4104, 0., 0.],
					[-8./27, 2., -3544./2565, 1859./4104, -11./40, 0.]]),
				(array([25./216, 0., 1408./2565, 2197./4104, -1./5, 0.]),
					array([
							16./135, 
							0., 
							6656./12825, 
							28561./56430, 
							-9./50, 
							2./55
						])
					)
				)
		}

		if method not in butcherArray:
			errorMessage = 'Method %s is not supported. Supported' \
				' methods are:\n'%method
			for key in butcherArray:
				errorMessage += '- %s\n'%key
			raise ValueError, errorMessage
		
		return butcherArray[method]
	
	def _get_method(self, method):
		"""Returns the class the method corresponds to."""
		methods = {
				'FE':ExplicitRungeKutta,
				'ET':ExplicitRungeKutta,
				'ERK2':ExplicitRungeKutta,
				'EM':ExplicitRungeKutta,
				'ERK3':ExplicitRungeKutta,
				'NYS':ExplicitRungeKutta,
				'ERK4':ExplicitRungeKutta,
				'ERK3/8':ExplicitRungeKutta,
				'BE':DiagonalImplicitRungeKutta,
				'IM':DiagonalImplicitRungeKutta,
				'SDIRK':DiagonalImplicitRungeKutta,
				'IT':DiagonalImplicitRungeKutta,
				'GCM4':ImplicitRungeKutta,
				'GCM6':ImplicitRungeKutta,
				'KB8':ImplicitRungeKutta,
				'EERK12':EmbeddedRungeKutta,
				'EERK23':EmbeddedRungeKutta,
				'EERK45':EmbeddedRungeKutta
		}

		return methods[method]

	def change_function(self, function):
		"""Changes the attribute function of strategy and context."""
		self.function = function
		try:
			self.rungeKutta.function = function
		except AttributeError:
			pass

		return None

	def change_initial_values(self, initialValues):
		"""Changes the attribute initialValues of strategy and context."""
		self.initialValues = initialValues
		try:
			self.rungeKutta.initialValues = initialValues
		except AttributeError:
			pass

		return None

	def build(self, method, verbose = False):
		"""
		
		"""
		A, b = self._get_RK_tableaux(method)
		self.rungeKutta = self._get_method(method)(
				A, 
				b, 
				self.function, 
				self.initialValues
			)

		if verbose:
			print 'Runge-Kutta method %s successfully built.'%method
			print 'The RK tableaux is given by:'
			print self.rungeKutta

		return None
	
	def integrate(
				self, 
				tStart, 
				tEnd, 
				number_of_steps,
				return_steps = False,
				flag = False, 
				verbose = False,
				**kwargs
			):
		"""

		"""
		
		try:
			tStart = float(tStart)
			tEnd = float(tEnd)
		except Exception as e:
			raise ValueError, str(e) + '. Initial time and end time must be' \
					' real numbers.'

		try:
			number_of_steps = int(number_of_steps)
		except Exception as e:
			raise ValueError, str(e) + '. The number of steps must be a ' \
					' natural number or at least be convertible to a' \
					' natural number.'
		
		#Switch tStart and tEnd in case of tEnd < tStart
		if tStart > tEnd:
			tmp = tStart
			tStart = tEnd
			tEnd = tStart
		elif tStart == tEnd:
			raise ValueError, 'Initial time and end time have to be' \
					' distinct.'

		t0 = time()

		h, t, y = self.rungeKutta.integrate(
							tStart,
							tEnd,
							number_of_steps,
							verbose,
							**kwargs
						)
		
		if verbose:
			print 'RK method was successfull in %d seconds.' %(time() - t0)
		
		if flag:
			return t[-1], y[:,-1]
		elif return_steps:
			return h, t, y
		elif return_steps and flag:
			return h, t[-1], y[:,-1]
		else:
			return t, y
