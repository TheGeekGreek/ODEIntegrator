# ODEIntegrator
ODEIntegrator is a class for solving first order explicit nonstiff ordinary differential equations using various Runge-Kutta methods. 

Installation
---
Download the six .py files. They are written on Python version 2.7.6. The main data-type is numpy.ndarray so numpy should be installed. Additionaly scipy.optimize will be used for solving non-linear equations.

Using ODEIntegrator
---
    instance = ODEIntegrator(right_hand_side, tstart, phi0)
    t, y = instance.integrate(tend)

Example
---
This can be found in the **docstring** of the class ODEintegrator.

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