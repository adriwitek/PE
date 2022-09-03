## Adrián Rubio Pintado
##13/06/2022

import numpy as np
import matplotlib.pyplot as plt
import stochastic_plots as stoch
from scipy import stats


def simula_trayectoria_discrete_mc(ini_state, P, t_ini,t_fin):
    """Simula 1 trayectorias para una cadena markok en tiempo 
        discreto

    Args:
        ini_state (int): initial state
        P(np matrix): transition matrix(P[i][j] is the prob. of going from i to state j)
        t_ini(int): initial time of simulation
        t_fin(int): final time of simulation
    Returns:
        tayectory(list): trajectory(seq. of states)

    """

    """
    Example
    -------
    >>> import numpy as np
    >>> import import codigo as pe

    >>> P = np.array([[0, 0.2, 0.8], 
                [0.3, 0, 0.7], 
                [0.6, 0.4, 0]])
    >>>  trajectory  = pe.simula_trayectoria_discrete_mc(0, P, 0,100)
    """


    trajectory = []
    trajectory.append(ini_state) 
    t = t_ini
    states = list(range( P.shape[0] ))
    state = ini_state



    while(True):
        #Generate state jumps with np.random.choice
        state = np.random.choice(states, p = P[state])
        # holding time is 1 for each state
        t += 1

        if t > t_fin:
            break

        trajectory.append(state)
        

    return trajectory




def simulate_continuous_time_Markov_Chain(P ,lambda_rates, state_0 , N_simul , t_ini=0 , t_fin=1000 ):

    """ Simula una CTMC (Continuous Time Markov Chain) 
    
    Args:
        P(np square matrix): transition matrix(P[i][j] is the prob. of going from i to state j)
        lambda_rates( np.ndarray) :
            Lambda rates of undergoing exponential waiting time distribution
        state_0(integer): 
           Init state
        N_simul(int) : 
            Number of simulations
        t_ini(float) : 
            Init time to start the simulation
        t_fin(float) : 
            Time in wich the simulation will end
            
    Returns:

        arrival_times(list): 
           List of list(one per simulation)with arrival times.Length of each list will be t1+1 due to t0 is added

        tajectories(list): 
            List of list(one per simulation) with trayectories(steps over states)
            Length of each list will be t1+1 due to init state is added
  

    Example
    -------
    >>> import codigo as pe
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> P =  np.array([[0, 0.2, 0.4, 0.4], 
                                      [0.3, 0, 0.5, 0.2], 
                                      [0.3, 0.4, 0, 0.3],
                                      [0.2, 0.3, 0.5, 0]
                                     ])

    >>> state_0 = 0 
    >>> lambda_rates = [1,1,1,1]
    >>> N_simul = 1 
    >>> N = 1000 
    >>> t_ini = 0 
    >>> t_fin = 1000 
    >>> times, trajectories = pe.simulate_continuous_time_Markov_Chain(P, 
    ...                                                              lambda_rates, 
    ...                                                              state_0,
    ...                                                              N_simul, 
    ...                                                              t_ini,
    ...                                                              t_fin)
    >>> fig, ax = plt.subplots(1, 1, figsize=(10,5), num=1)
    >>> ax.step(times[0], trajectories[0],where='post')
    >>> ax.set_ylabel('State ')
    >>> ax.set_xlabel('Time ')
    >>> _ = ax.set_title('Simulation result of a Continuos Time Markov chain')
    """

    states = list(range( P.shape[0] ))

    # Outputs
    arrival_times = []
    trajectories = []


    for _ in range(N_simul):

        # Init each simulation(trajectory + holding times). 
        state = state_0
        t = t_ini

        currentsim_times = [t_ini]
        currentsim_trajectory = [state_0]
        
        while(1):

            # Get holding time 
            t += np.random.exponential(scale=1.0/lambda_rates[state])
        
            # If time is greater than end time, finish
            if t > t_fin:
                break

            state = np.random.choice(states, p = P[state])
            currentsim_times.append(t)
            currentsim_trajectory.append(state)

        # Saving list with the current sim results 
        arrival_times.append(currentsim_times)
        trajectories.append(currentsim_trajectory)

    return arrival_times, trajectories



def ode_euler(t0, x0, T, a, N):
    """ Integration of an ODE using the Euler scheme
		Author: Alberto Suárez

        x(t0) = x0
        dx(t) = a(t, x(t))*dt   

    Args:
    
        t0 : float
            Initial time for the simulation
        x0 : float
            Initial level of the process
        T : float
            Length of the simulation interval [t0, t0+T]
        a :
            Function a(t,x(t)) that characterizes the drift term
        N: int
            Number of intervals for the simulation

    Returns
    
        t: numpy.ndarray of shape (N+1,)
            Regular grid of discretization times in [t0, t0+T]
        X: numpy.ndarray of shape (1,N+1)
            Simulation consisting of 1 trajectories.
            Each trajectory is a row vector composed of the values
            of the process at t.
    """

    dT = T / N # size of simulation step
    
    # Initialize solution array
    t = np.linspace(t0, t0+T, N+1) # integration grid
    x = np.zeros(N+1)
    
    # Initial condition
    x[0] = x0
 
    # Integration of the ODE
    for n in range(N):
        x[n+1] = x[n] + a(t[n], x[n])*dT
        
    return (t, x)


    

def euler_maruyana(t0, x0, T, a, b, M, N):
    """ Numerical integration of an SDE using the stochastic Euler scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)  (Itô SDE)

    Args:
    
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
    
        t: numpy.ndarray of shape (N+1,)
            Regular grid of discretization times in [t0, t0+T]
        X: numpy.ndarray of shape (M,N+1)
            Simulation consisting of M trajectories.
            Each trajectory is a row vector composed of the values
            of the process at t.
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


#Aux function
def dW(delta_t, M):
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t), size = M)

def milstein(t0, x0, T, a, b, db_dx, M, N):
    """ Numerical integration of an SDE using the stochastic Euler scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)  (Itô SDE)

    Args:
    
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
        db_dx: 
            Function db_dx(t,x(t)), derivate of db respct to dx
        M: int
            Number of trajectories in simulation
        N: int
            Number of intervals for the simulation

    Returns
    
        t: numpy.ndarray of shape (N+1,)
            Regular grid of discretization times in [t0, t0+T]
        X: numpy.ndarray of shape (M,N+1)
            Simulation consisting of M trajectories.
            Each trajectory is a row vector composed of the values
            of the process at t.
    """
    dt = T/N  # size of simulation step

    # Initialize solution array
    t = np.linspace(t0, t0 + T, N + 1)  # integration grid
    X = np.zeros((M, N + 1))

    # Initial condition
    X[:, 0] = np.full(M, x0)
    
    
    for i in range(N):
        X[:, i + 1] = X[:, i] + a(t[i], X[:, i])*dt + b(t[i], X[:, i])*dW(dt,M)+  0.5*db_dx(t[i], X[:, i])* (dW(dt,M)**2 - dt)
        
    return t, X
