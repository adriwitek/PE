# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:29:26 2020

@author: Alberto Suárez
"""
# Load packages
import numpy as np


def euler_maruyana(t0, x0, T, a, b, M, N):
    """ Numerical integration of an SDE using the stochastic Euler scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t,x(t)) that characterizes the drift term
    b :
        Function b(t,x(t)) that characterizes the diffusion term
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values
        of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> t, S = sde.euler_maruyana(t0, S0, T, a, b, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Euler scheme)')

    """

    dt = T / N # size of simulation step
    
    # Initialize solution array
    t = np.linspace(t0, t0+T, N+1) # integration grid
    X = np.zeros((M,N+1))
    
    #Initial condition
    X[:, 0] = np.full(M, x0)
    
    for i in range(N): 
        X[:, i+1] =  X[:, i] + a(t[i], X[:, i]) * dt + b(t[i], X[:,i]*np.random.normal(loc=0.0, scale=np.sqrt(dt), size=M))
                                                        
    return t, X


def dW(delta_t, M):
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t), size = M)
def b2(t,x): return sigma**2*x

def milstein(t0, x0, T, a, b, db_dx, M, N):
    """ Numerical integration of an SDE using the stochastic Milstein scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t, x(t)) that characterizes the drift term
    b :
        Function b(t, x(t)) that characterizes the diffusion term
    db_dx:
        Derivative wrt the second argument of b(t, x)
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> def db_dSt(t, St): return sigma
    >>> t, S = sde.milstein(t0, S0, T, a, b, db_dSt, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Milstein scheme)')

    """

def milstein(t0, x0, T, a, b, db_dx, M, N):
    dt = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)
    
    
    for i in range(N):
        X[:, i + 1] = X[:, i] + a(t[i], X[:, i])*dt + b(t[i], X[:, i])*dW(dt,M)+  0.5*b2(t[i], X[:, i])* (dW(dt,M)**2 - dt)
        
    return t, X

def simulate_jump_process(t0, T, simulator_arrival_times, simulator_jumps, M):
    """ Simulation of jump process

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    T : float
        Length of the simulation interval [t0, t0+T]
    simulator_arrival_times: callable with arguments (t0,T)
        Function that returns a list of M arrays of arrival times in [t0, t0+T]
    simulator_jumps: callable with argument N
        Function that returns a list of M arrays with the sizes of the jumps
    M: int
        Number of trajectories in the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    """

    times_of_jumps = [[] for _ in range(M)]
    sizes_of_jumps = [[] for _ in range(M)]
    for m in range(M):
        times_of_jumps[m] = simulator_arrival_times(t0, T)
        max_jumps = len(times_of_jumps[m])
        sizes_of_jumps[m] = simulator_jumps(max_jumps)
    return times_of_jumps, sizes_of_jumps

"""
    time_of_jumps =  list(map(lambda _: simulator_arrival_times(t0, T), list(range(M))))
    sizes_of_jumps = list(map(lambda x: simulator_jumps(len(x)), time_of_jumps))
    return time_of_jumps, sizes_of_jumps
"""

# Stochastic Euler scheme for the numerical solution of a jump-diffision SDE
def euler_jump_diffusion(t0, x0, T, a, b, c,
                         simulator_jump_process,
                         M, N):
    """ Simulation of jump diffusion process

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t) + c(t, x(t)) dJ(t)

    [Itô SDE with a jump term]


    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a : Function a(t,x(t)) that characterizes the drift term
    b : Function b(t,x(t)) that characterizes the diffusion term
    c : Function c(t,x(t)) that characterizes the jump term
    simulator_jump_process: Function that returns times and sizes of jumps
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t
    """

    dt = T/N  # size of simulation step
    
    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))
    # Initial condition
    X[:, 0] = np.full(M, x0)
    
    
    times_of_jumps, sizes_of_jumps = simulator_jump_process(t0, T, M)
    

    # Para cada simulacion
    for n in range(N):

        # Integrate from t to t+\inc T
        integrate_times = [(tau>t[n])*(tau<t[n+1]) for tau in times_of_jumps]

        # Donde hay saltos
        inside_time_jumps = list(map(lambda x: bool(x.sum()), integrate_times))
        
        tau_ = np.full(M, t[n])
        Xtau_ = X[:, n]

        # Intervalos donde hay salto
        for m in np.where(inside_time_jumps)[0]:
            tau, Y  = times_of_jumps[m][integrate_times[m]], sizes_of_jumps[m][integrate_times[m]]
            Z, dT_jump = np.random.randn(), tau - tau_[m]
            Stau_ = Xtau_[m] + a(tau_[m],Xtau_[m])*dT_jump + b(tau_[m], Xtau_[m])*np.sqrt(dT_jump)*Z
            tau_[m] = tau[-1]
            Xtau_[m] = Stau_[-1] + c(tau_[m], Stau_[-1])*Y[-1]

        # Paso siguiente
        Z = np.random.randn(M)
        dT_jump = t[n + 1] - tau_
        X[:, n + 1] = Xtau_ + a(tau_, Xtau_)*dT_jump + b(tau_, Xtau_)*np.sqrt(dT_jump)*Z
        
    return t, X
