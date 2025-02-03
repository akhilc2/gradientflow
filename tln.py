import numpy as np 
from scipy.linalg import expm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathos.multiprocessing import ProcessPool
from numba import njit, jit
import gc
#computing all the wavevectors first

@njit
def wave_vectors(Ns, Nt):
    lattice = np.zeros((Ns, Ns, Ns, Nt, 4)) #Our lattice where each point has a wavevector 
    indices = np.zeros((Ns, Ns, Ns, Nt, 4))
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    indices[x,y,z,t] = [x, y, z, t]
                    lattice[x,y,z,t,:] = 2*np.pi * np.array([x,y,z,t])/np.array([Ns, Ns, Ns, Nt]) 
    return lattice, indices


#Some helper functions for the momentum

def p_hat(n):
    return 2*np.sin(lattice[tuple(n)]/2)

def p_tilde(n):
    return np.sin(lattice[tuple(n)])

def p_hat_sq(n):
    return np.sum(p_hat(n)**2)

def p_tilde_sq(n):
    return np.sum(p_tilde(n)**2)

def delta(mu,nu):
    return 1 if mu == nu else 0 

def inverse(M):
    return np.linalg.inv(M)

#Here we define the Symanzik/Clover action 

def Action(n,mu,nu,c): #c here will determine if its the gauge action, flow action, or the operator to compute E(t)
    S = delta(mu,nu) * (p_hat_sq(n) - c * np.sum(p_hat(n)**4) - c * (p_hat(n)**2)[mu] * p_hat_sq(n)) - p_hat(n)[mu]*p_hat(n)[nu]*(1 - c * ((p_hat(n)**2)[mu] + (p_hat(n)**2)[nu]) )
    return S

def Clover(n, mu, nu):
    K = (delta(mu,nu)*p_tilde_sq(n) - p_tilde(n)[mu]*p_tilde(n)[nu]) * np.cos(lattice[tuple(n)][mu]/2)*np.cos(lattice[tuple(n)][nu]/2)
    return K

def gauge_fix(n, mu, nu, alpha):
    G = (p_hat(n)[mu]*p_hat(n)[nu])/alpha
    return G

action = np.vectorize(Action, otypes= [np.ndarray], signature="(n),(),(),()->()") #Doing this vectorization doesnt really work well. Probably because of how im implementing it
g = np.vectorize(gauge_fix, otypes= [np.ndarray], signature="(n),(),(),()->()")

def create_LatSites(indices, cf, cg, ce):
    alpha = 1
    Sf = np.zeros((len(indices), 4, 4))
    Sg = np.zeros((len(indices), 4, 4))
    Se = np.zeros((len(indices), 4, 4))
    G = np.zeros((len(indices), 4, 4))
    Inv = np.zeros((len(indices), 4, 4))
    for n in range(len(indices)):
        index = np.array(indices[n], dtype = int)
        for mu in range(4):
            for nu in range(4):
                Sf[n, mu, nu] = Action(index, mu, nu, cf)
                G[n, mu, nu] = gauge_fix(index, mu, nu, alpha)
                if cg != cf and cg != 999:
                    Sg[n, mu, nu] = Action(index, mu, nu, cg)
                elif cg == 999:
                    Sg[n,mu, nu] = Clover(index, mu, nu)
                if ce != cf  and ce != cg and ce != 999:
                    Se[n, mu, nu] = Action(index, mu, nu, ce)  #I hope to have covered all the cases here 
                elif ce == 999:
                    Se[n, mu, nu] = Clover(index, mu, nu)
    if cf == cg:
        Sg = Sf.copy()
    if cf == ce:
        Se = Sf.copy()
    if cg == ce:
        Se = Sg.copy()


    Sg_plus_G = Sg + G
    Sf_plus_G = Sf+G
    Inv = inverse(Sg_plus_G)
    return Sf_plus_G, Inv, Se 

def create_lattice_parallel(indices, cf, cg, ce, num_batches = 4):  #Need to figure out how many batches work the best. Ns/4 seems to work? Get different answers depending on the batch size
    batches = np.array_split(indices, num_batches)
    
    with ProcessPool(nodes=num_batches) as executor:
        results = list(executor.map(lambda batch: create_LatSites(batch, cf, cg, ce), batches))
    
    Sf_plus_G_total = np.concatenate([result[0] for result in results])
    Inv_total = np.concatenate([result[1] for result in results])
    Se_total = np.concatenate([result[2] for result in results])

    return Sf_plus_G_total, Inv_total, Se_total
###This cell is to run the code


def tln_batch_parallel(flowtimes, Sf_plus_G, Inv, Se, num_batches = 4):
    # Split the flowtimes into batches
    batches = np.array_split(flowtimes, num_batches)
    
    # Function to process each batch
    def process_batch(batch):
        batch_results = []
        for t_flow in batch:
            prefactor = 128 * np.pi**2 * t_flow**2 / (3 * (Ns**3*Nt))
            otherfactor = 64 * np.pi**2 * t_flow**2 / (3 * Ns**3*Nt)
            Exp = expm(-t_flow*(Sf_plus_G))
            trace  = np.trace(Exp @ Inv @ Exp @ Se, axis1 = 1, axis2 = 2)
            sum = np.sum(trace) #here is the most expensive part (I think)
            batch_results.append( prefactor + otherfactor*sum)
            del Exp, sum, trace
            gc.collect()
        return batch_results
    
    # Use ThreadPoolExecutor to process batches in parallel
    with ProcessPool(nodes=num_batches) as executor:
        results = list(executor.map(process_batch, batches))
    
    final_output = np.concatenate(results)
    return final_output


def compute_tln(Ns, Nt, flow ="W", action = "W", operator = "W"):
    eps = 0.01
    n_t = int(np.ceil(Ns**2 / 32 / eps) + 1)  # number of flow times
    flowtimes = np.arange(n_t)* eps
    
    cf, cg, ce = 0, 0, 0 #Flow is always wilson. Will need to be changed to make general
    if action == "S":
        cg = -1/12
    elif action == "C":
        cg = 999 #Im trying to minimize if statements so I am giving the clover action a number to indentify 
    if operator == "S":
        ce = -1/12
    elif operator == 'C':
        ce = 999
    
    #first we create our lattice here (without using symmetries)

    lattice, indices = wave_vectors(Ns, Nt)
    indices = np.int64(indices.reshape(-1,4)[1:]) #ommitting the n^2 = 0 vector and flattening this.
    
    S_plus_G, Inv, Se = create_lattice_parallel(indices, cf, cg, ce, num_batches= Nt//2)
    tln = tln_batch_parallel(flowtimes, S_plus_G, Inv, Se, num_batches= Nt//2)

    #Then we compute all of our matrices using the above function
        
    return tln

#Run this cell
Ns, Nt = 16,16
flow, action, operator = 'W', 'S', 'C'
lattice, indices = wave_vectors(Ns,Nt)

#Tested everything and its correct!
compute_tln(Ns, Nt, flow=flow, operator=operator,action=action)
