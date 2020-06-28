import matplotlib.pyplot as plt 
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign

def gen_ang(numproj, nProj_per_rot):
    """Generate interlaced angles"""

    prime = 3
    pst = 0
    pend = 360
    seq = []
    i = 0
    sgn = 1  # for switching direction
    while len(seq) < numproj:
        b = i
        i += 1
        r = 0
        q = 1 / prime
        while (b != 0):
            a = np.mod(b, prime)
            r += (a * q)
            q /= prime
            b = np.floor(b / prime)
        r *= ((pend-pst) / nProj_per_rot)
        k = 0
        while (np.logical_and(len(seq) < numproj, k < nProj_per_rot)):
            if(sgn == 1):
                seq.append(pst + (r + k * (pend-pst) / nProj_per_rot))
            else:
                seq.append(pend-((1-r) + k * (pend-pst) / nProj_per_rot))
            k += 1
    return seq

if __name__ == "__main__":

    n = 512
    nz = n
    ntheta = n*3//2
    pprot = n*3//4
    
    data = np.ones([ntheta,nz,n],dtype='float32')
    theta = np.array(gen_ang(ntheta, pprot)).astype('float32')

    center = n/2
    pnz = 16 # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    ptheta = 16 # number of projections for simultaneus processing by one GPU in the alignment sub-problem    
    ngpus = 4 # number of gpus
    niteradmm = 128 # number of iterations in the ADMM scheme
    
    uof = tomoalign.admm_of(data, theta, pnz, ptheta , center, ngpus, niteradmm)
    
   