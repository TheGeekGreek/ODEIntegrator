# ODEIntegrator

ODEIntegrator is a class for solving first order explicit nonstiff ordinary differential equations using various Runge-Kutta methods. 

Installation
---
Download the six .py files. They are written on Python version 2.7.6. The main data-type is numpy.ndarray so numpy should be installed. Additionaly scipy.optimize will be used for solving non-linear equations.

Using ODEIntegrator
---
    instance = ODEIntegrator(right_hand_side, tstart, phi0)
    t, y = instance.integrate(tend)
