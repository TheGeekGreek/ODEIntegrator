"""
ODEIntegrator - A class for integrating systems of first order explicit 
nonstiff ordinary differential equations using various Runge Kutta 
methods. Implemented are explicit, semi-implicit, implicit and 
embedded methods up to the order eight.

Author:
-------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from numpy import array, ndarray, sqrt, finfo, tile
from time import time
from types import FunctionType

#Wildcard import from own modules
from ExplicitRungeKutta import *
from DiagonalImplicitRungeKutta import *
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
	N_{>0} and a given Cauchy-problem of the form y(xstart) = 
	y_0 in R^n on the interval [xstart, xend] in R.

	Attributes:
	----------
	function : function
			The function f given in y'(x) = f(x,y(x)).

	tstart : float
			The initial time given in the Cauchy-problem.
	
	initial_value: numpy.ndarray
			The initial value provided by the Cauchy-problem. 

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
	>>> initial_value = array([-pi/3,0])
	>>> tStart = 0
	>>> tEnd = 10
	>>> instance = ODEIntegrator(right_hand_side, tStart, initial_value)
	Runge-Kutta method EERK45 successfully built.
	>>> h, t, y = instance.integrate(
	...     tEnd,
	...     return_steps = True
	... )
	>>> print h
	[ 0.02770747  0.05541494  0.11082988  0.22165976  0.27230999  0.28052346
	  0.27419695  0.24301475  0.2163588   0.200596    0.20012012  0.21079332
	  0.22729721  0.25175401  0.27977788  0.28172309  0.26824431  0.26540465
	  0.27527766  0.28159341  0.26379881  0.23171046  0.20941502  0.19673453
	  0.20345275  0.21603736  0.23521117  0.26258503  0.28467372  0.27689268
	  0.26364826  0.26899637  0.27857908  0.27905196  0.25179636  0.22226502
	  0.20393565  0.19810893  0.20745327  0.22220389  0.24437379  0.27333594
	  0.26114229]
	>>> print t
	[  0.           0.02770747   0.08312241   0.19395229   0.41561205
	   0.68792204   0.9684455    1.24264245   1.48565721   1.702016     1.902612
	   2.10273211   2.31352544   2.54082265   2.79257666   3.07235453
	   3.35407762   3.62232193   3.88772657   4.16300424   4.44459764
	   4.70839645   4.94010691   5.14952193   5.34625645   5.54970921
	   5.76574657   6.00095774   6.26354278   6.54821649   6.82510918
	   7.08875744   7.35775381   7.63633288   7.91538484   8.16718121
	   8.38944623   8.59338189   8.79149082   8.99894408   9.22114798
	   9.46552177   9.73885771  10.        ]
	>>> print y
	[[-1.04719755 -1.04686514 -1.04420658 -1.03093436 -0.97295233 -0.84656229
	  -0.65872635 -0.42888388 -0.19876298  0.01626545  0.21517053  0.40510363
	   0.58807394  0.75772866  0.90418737  1.00859586  1.04706807  1.02003023
	   0.93313189  0.78355773  0.57568896  0.34186411  0.11687685 -0.09214042
	  -0.28501638 -0.47302027 -0.65198673 -0.81461813 -0.94864149 -1.03079846
	  -1.04428336 -0.9956984  -0.88625884 -0.71424335 -0.49139217 -0.25861052
	  -0.03929693  0.16388796  0.3548873   0.54025377  0.71422566  0.86819875
	   0.98646648  1.04152938]
	 [ 0.          0.02399384  0.07194461  0.16743713  0.35457516  0.57019657
	   0.76259136  0.90491009  0.98011491  0.99986897  0.97666923  0.91549225
	   0.81487697  0.67290479  0.48646879  0.25710238  0.01508841 -0.21606491
	  -0.43662892 -0.64561445 -0.82318856 -0.94035531 -0.99315725 -0.99575223
	  -0.95881211 -0.88340221 -0.76796257 -0.61016324 -0.40691752 -0.16815246
	   0.07106686  0.29638046  0.51442899  0.7149728   0.873706    0.96617899
	   0.9992326   0.98651455  0.93561832  0.84567535  0.71499078  0.54075401
	   0.32139002  0.09905591]]
	"""
	def __init__(self, function, tstart, initial_value):
		"""
		Initializer of an instance of the class ODEIntegrator.
		
		Does some basic testing on the input provided by the user.

		Parameters:
		----------
		function : FunctionType
				Right-hand side of the ODE.

		tstart : float
				Initial time provided by the Cauchy-problem.

		initial_value : array_like
				Initial value provided by the Cauchy-problem.
		"""
		if isinstance(function, FunctionType):
			self.function = function
		else:
			raise ValueError, 'Expected an object of type function' \
					' got %s instead.'%type(function)

		try:
			self.tstart = float(tstart)
		except Exception as e:
			raise ValueError, str(e) + '. Starting time has to be a' \
					' float.'

		if isinstance(initial_value, ndarray):
			if initial_value.ndim > 1 or initial_value.size == 0:
				raise ValueError, 'Initial value has to be a non-' \
						' empty vector not a tensor. ndim = %s'%initial_value.ndim
			else:
				self.initial_value = initial_value
		else:
			raise ValueError, 'Initial value has to be a numpy' \
					' ndarray got %s instead.'%type(initial_value)

		#Check if function and initial value matches
		val = function(tstart, initial_value)
		
		if isinstance(val, ndarray):
			if val.shape != initial_value.shape:
				raise ValueError, 'Function has to be vector-' \
						' valued with same number of entries as' \
						' the initial value.'
		else:
			raise ValueError, 'The value of the function has to' \
					' be of type numpy ndarray.'
		
		#Default settings
		self.settings = {
				'method':str('EERK45'),
				'steps':int(),
				'params':(),
				'verbose':False,
				'flag':False,
				'return_steps':False,
				'abstol':tile(array([1e-6]), self.initial_value.size),
				'reltol':tile(array([1e-6]), self.initial_value.size),
				'h_min':finfo(float).eps,
				'h_max':float(),
				'max_steps':int(1e+4)
			}

		return None	

	def _get_rk_tableaux(self, method):
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
        Method Order Full name					   Reference
        ====== ===== ============================= ==================
        FE     1     Forward Euler				   [Ise96]_, page 4 
        ET     2     Explicit trapezoidal or Heun  [Ise96]_, page 39
        ERK2   2     Ralston					   [Ise96]_, page 39
        EM     2     Explicit midpoint			   [Ise96]_, page 39
		ERK3   3     Classical RK order 3		   [Ise96]_, page 40
		NYS    3	 Nystrom					   [Ise96]_, page 40
		SDIRK  3     Singly diagonal implicit	   [Hai00]_, page 207
        ERK4   4     "The" Runge-Kutta method	   [Hai00]_, page 138
        ERK3/8 4     3/8-rule					   [Hai00]_, page 138 
        BE     1     Backward Euler				   [Hai00]_, page 205
        IM     2     Implicit Midpoint rule		   [Hai00]_, page 205 
        IT     2     Implicit trapezoidal		   Wikipedia* 
        GL4    4     Gauss-Legendre				   [Ise96]_, page 47
        GL6    6     Gauss-Legendre				   [Ise96]_, page 47
		KB8    8     Kuntzmann & Butcher           [Hai00]_, page 209
		EERK12 2(1)  Heun-Euler					   Wikipedia*
        EERK23 3(2)  Simple embedded RK pair	   [Ise96]_, page 84
		EERK45 5(4)  Fehlberg                      [Ise96]_, page 84
        ====== ===== ============================= ==================

		* List of Runge-Kutta methods.

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
					[-8./27, 2., -3544./2565, 1859./4104, -11./40, 0.]
				]),
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
			self.runge_kutta.function = function
		except AttributeError:
			pass

		return None

	def change_initial_time(self, tstart):
		"""Changes the attribute tstart of context."""
		ODEIntegrator.__init__(
				self, 
				self.function, 
				tstart, 
				self.initial_value
			)
		
		return None

	def change_initial_value(self, initial_value):
		"""Changes the attribute initial_value of strategy and context."""
		self.initial_value = initial_value
		
		try:
			self.runge_kutta.initial_value = initial_value
		except AttributeError:
			pass

		return None

	def _build(self, method):
		"""Calls __init__ of RungeKutta and generates attribute."""
		rk_matrix, rk_weights = self._get_rk_tableaux(method)
		
		#Generating of new attribute
		self.runge_kutta = self._get_method(method)(
				rk_matrix, 
				rk_weights, 
				self.function, 
				self.initial_value,
				self.settings
			)

		print 'Runge-Kutta method %s successfully built.'%method

		return None

	def _process_kwargs(self, kwdict):
		"""Evaluates and saves optional settings provided by the user."""
		for key in kwdict:
			if key in self.settings:
				if type(kwdict[key]) != type(self.settings[key]):
					raise ValueError, 'Parameter %s in settings has to be ' \
							' of type %s. Got %s instead'%(
									key, 
									type(self.settings[key]),
									type(kwdict[key])
								)
				else:
					self.settings[key] = kwdict[key]
			else:
				raise ValueError, 'Not supported keyword argument %s ' \
						' provided.'%key 
		
		#If method is non-adaptive it is mandatory to have non-zero steps 
		if not (self.settings['steps'] == int()):
			if not (self._get_method(self.settings['method']) == EmbeddedRungeKutta):
				raise ValueError, 'The keyword argument steps has to be different' \
						' from zero for the non-adaptive method %s.'%self.settings['method']

		return None
	
	def integrate(self, tend, **kwargs):
		"""
		Integrates a given initial value problem for a given endpoint.

		Parameters:
		----------
		tend : float
				Endpoint of integration.

		method : string, optional
				Method described in the docstring of _get_rk_tableaux.

		steps : int, semioptional
				If method is changed from default steps indicates how many
				steps are taken for integration.

		params : tuple, optional
				Additional arguments of the function passed in *args.

		verbose : bool, optional
				If True statistics about integration are printed on console.

		flag : bool, optional
				If True the integrator only returns the value at tend.

		return_steps : bool, optional
				If True returns the stepsize for non-addaptive method or
				an array of the steps for an adapive method.

		abstol : array_like, optional
				Array of the absolute tolerance for adaptive method.

		reltol : arra_like, optional
				Array of the relative tolerance for adaptive method.

		h_min : float, optional
				Minimal stepsize for adaptive mezthod.

		h_max : float, optional
				Maximal stepsize for adaptive method.

		max_steps : int, optional
				Maximal number of steps for adaptive method.

		Returns:
		-------
		h : array_like/float
				Array of steps for adaptive method float else.

		t : array_like
				Equidistant or non-equidistant [tstart, tend].

		y : array_like
				Solution of the initial value problem. This is a matrix
				of shape initial_value.size x time.size.
		"""
		try:
			tend = float(tend)
		except Exception as e:
			raise ValueError, str(e) + '. End time has to be a float.'
		
		#Switch tStart and tEnd in case of tEnd < tStart
		if self.tstart > tend:
			tmp = self.tstart
			self.tstart = tend
			tend = tmp
		
		elif self.tstart == tend:
			raise ValueError, 'Initial time and end time have to be' \
					' distinct.'
		
		#Default setting
		self._process_kwargs(kwargs)
		self._build(self.settings['method'])
		
		if self.settings['h_max'] == float():
			self.settings['h_max'] = float(tend - self.tstart)
			
		#Initialize time measurement of integration process
		t0 = time()
		
		#Initialize integration in the corresponding class
		h, t, y = self.runge_kutta.integrate(
							self.tstart,
							tend
						)
		
		if self.settings['verbose']:
			print 'Integration process took %f seconds.'%(time() - t0)
		
		if self.settings['flag']:
			return t[-1], y[:,-1]
		elif self.settings['return_steps']:
			return h, t, y
		elif self.settings['return_steps'] and self.settings['flag']:
			return h, t[-1], y[:,-1]
		else:
			return t, y
