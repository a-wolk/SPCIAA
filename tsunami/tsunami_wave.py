import matplotlib.pyplot as plt
import numpy as np 
import itertools, copy, sys
import pandas
np.set_printoptions(threshold=sys.maxsize)

def calculate_wave(i, j, u_t_1, u_t_2, b_t, t, C, g=9.81):
    h = 10/u_t_1.shape[0]
    X_1 = ((u_t_1[i+1, j] - u_t_1[i-1, j]) / 2*h)**2
    X_2 = (u_t_1[i, j] - b_t[i, j]) * (
        (u_t_1[i+1, j]-2*u_t_1[i, j]+ u_t_1[i-1, j]) / h**2
    )
    X_3 = ((u_t_1[i, j+1] - u_t_1[i, j-1]) / 2*h)**2
    X_4 = (u_t_1[i, j] - b_t[i, j]) * (
        (u_t_1[i, j+1]-2*u_t_1[i, j]+ u_t_1[i, j-1]) / h**2
    )

    X = X_1 + X_2 + X_3 + X_4
    
    val = u_t_1[i, j] + C * (u_t_1[i, j] - u_t_2[i, j]) + g * (t**2) * X
    return val

def wave_step(center, u_t_1, u_t_2, b_t, t, C, g=9.81):
   x, y = center
   size = u_t_1.shape[0]
   u_new = np.full(shape = (size,size), fill_value = 2, dtype = float)

   for i in range(1, size-1):
      for j in range(1, size-1):
         u_new[i, j] = calculate_wave(i, j, u_t_1, u_t_2, b_t, t, C, g)
         
        #  r = np.sqrt((i-x)**2 + (j-y)**2)
        #  u_new[i, j] += (t*t*(2 + 2 * np.exp(-(r**2))))
        #  u_new[i, j] += (t*t*2)   
   
   return u_new, u_t_1

def create_board(size, center, water_lvl):
    x, y = center
    delta = 10 / size
    x *= delta
    y *= delta
    board = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i*delta-x)**2 + (j*delta-y)**2)
            board[i, j] = 2 + 2 * np.exp(-(r**2))

    return board

def wave_simulation(size, center, iterations, t, C, b_t=None, water_lvl=2):
    u_t_1 = create_board(size, center, water_lvl)
    u_t_2 = create_board(size, center, water_lvl)
    b_t = np.zeros((size, size)) if b_t is None else b_t

    import time
    import pylab as pl
    from IPython import display
    
    # plt.figure()
    # plt.colorbar()
    for i in range(iterations-1):
        u_t_1, u_t_2 = wave_step(center, u_t_1, u_t_2, b_t, t, C)

        if i % 10 == 0:
          # plt.figure()
          # plt.imshow(u_t_1)
          
          # plt.colorbar()
          # display.clear_output(wait=True)
          # display.display(pl.gcf())

          np.save(f"out_{i}.data", u_t_1)
        
    return u_t_1