import numpy as np
gridworld = np.zeros(16).reshape(4,4)

def update(i,j,gridworld):
    if i+j == 0:
        return 0
    if i+j == 6:
        return 0
    
    n = gridworld[i,j-1 if j > 0 else 0] -1
    s = gridworld[i,j+1 if j < 3 else 3] -1 
    w = gridworld[i-1 if i > 0 else 0,j] -1 
    e = gridworld[i+1 if i < 3 else 3,j] -1
    
    # Termination states
    w = 0 if i == 1 and j == 0 else w
    n = 0 if i == 0 and j == 1 else n
    s = 0 if i == 3 and j == 2 else s
    e = 0 if i == 2 and j == 3 else e
        
    retval = (n+s+e+w)/4
    return retval

def value_iterate(eps=0.01,nmax=100):
    gridworld = np.zeros(16).reshape(4,4)
    run = 0
    error = 1.0
    while error > eps:
        run = run + 1
        
        newgrid =  np.zeros(16).reshape(4,4)
        for i in range(4):
            for j in range(4):
                newgrid[i,j] = update(i,j,gridworld)
        delta = newgrid - gridworld
        error = np.abs(delta).max()
        print(f'Step {run}: {error}')
        gridworld = newgrid
    return gridworld


        
gridworld = value_iterate(eps=0.01)
