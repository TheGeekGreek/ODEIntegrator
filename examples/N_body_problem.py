# -*- coding: utf-8 -*-
"""
Author:
------
Yannis Baehni - yannis.baehni@uzh.ch

at University of Zurich, Raemistrasse 71, 8006 Zurich.
"""
from numpy import array, zeros_like, zeros, sqrt, log, tile, hstack, linspace
from numpy.linalg import norm
from numpy.random import rand, seed
from matplotlib.pyplot import plot, show, savefig, grid, figure, legend 
from matplotlib.pyplot import xlim, xlabel, ylabel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from ODEIntegrator import *

def momentum(y, k, i):
	N = y.shape[0]/6.
	denom = y[3 * (N + i):3 * (N + i + 1)] - y[3 * (N + k):3 * (N + k + 1)]
	nom = norm(denom)**3
	return denom/nom

def hamiltonian_system(t, y, N):
	val = zeros_like(y)
	val[3 * N:] = y[:3 * N]
	for k in range(N):
		val[3 * k:3 * (k+1)] = sum([momentum(y, k, i) for i in range(N) if i != k])
	return val

def speed(y, k, i):
	N = y.shape[0]/6.
	return norm(y[3 * (N + k):3 * (N + k + 1),i])

def update_lines(num, dataLines, lines):
	ax.view_init(30, .3 * num)
	num = (2 * num) % y.shape[1]
	for index, line in enumerate(lines):
		line.set_data(dataLines[3 * index:3 * index + 2, :num])
		line.set_3d_properties(dataLines[3 * index + 2, :num])
	return lines

def get_coordinates(matrix, i):
	out = [y[3 * k + i:3 * (k + 1) + i,:] for k in range(matrix.shape[0]/6)]
	return array(out).flatten()


if __name__ == '__main__':
	N = 6
	tstart = 0.
	tend = 1.5
	
	seed(56857)
	y0 = rand(6 * N)

	instance = ODEIntegrator(hamiltonian_system, tstart, y0)
	t, y = instance.integrate(tend, params = (N,), verbose = True)
	
	fig = figure(figsize = (15, 10))
	ax = fig.add_subplot(111, projection = '3d')	
	
	for k in range(N):
		lab = r'$B_{%s}$'%(k+1)
		ax.plot(y[3 * k,:], y[3 * k + 1,:], y[3 * k + 2,:], label = lab)
	
	legend()
	savefig('N_body_problem_trajectories.pdf')
	show()

	fig = figure(figsize = (15, 10))
	n = len(t)

	for k in range(N):
		plot(t, array([speed(y, k, i) for i in range(n)]), 'x-')
	
	legend()
	xlabel(r'$t$')
	ylabel(r'$\|p_k\|$')
	xlim(0, tend)
	grid(True)
	savefig('N_body_problem_velocities.pdf')
	show()

	##########################################################################
	#Animation															 	 #
	##########################################################################
	fig = figure(figsize = (15, 10))
	ax = fig.add_subplot(111, projection = '3d')
	
	x1 = get_coordinates(y, 0)
	x2 = get_coordinates(y, 1)
	x3 = get_coordinates(y, 2)

	ax.set_xlim3d([min(x1), max(x1)])
	ax.set_ylim3d([min(x2), max(x2)])
	ax.set_zlim3d([min(x3), max(x3)])

	lines = [ax.plot(y[3 * k, 0:1], y[3 * k + 1, 0:1], y[3 * k + 2, 0:1])[0] for k in range(N)]
	
	anim = FuncAnimation(
			fig, 
			update_lines, 
			fargs = (y,lines), 
			interval = 1, 
			blit = False
	)

	#anim.save('N_body.mp4', fps=15, writer = 'mencoder')
	show()
