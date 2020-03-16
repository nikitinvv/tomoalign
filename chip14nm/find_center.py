import dxchange
import numpy as np
import tomocg as tc
import scipy as sp
import sys
import os



if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    prj = np.load('/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/prjbin2.npy')[0:ndsets*100].astype('complex64')                                 
    theta = np.load('/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/theta.npy')[0:ndsets*100].astype('float32')

    # data
    data = prj[:,prj.shape[1]//2:prj.shape[1]//2+1].copy()
    data[np.isnan(data)]=0
    print(np.linalg.norm(data))
    [ntheta, nz, n] = data.shape  # object size n x,y
    
    center = 1250
    binning = 2
    niter = 1  # tomography iterations
    pnz = 1  # number of slice partitions for simultaneous processing in tomography    

    # initial guess
    u = np.zeros([nz, n, n], dtype='complex64')
    psi = data.copy()
    lamd = np.zeros([ntheta, nz, n], dtype='complex64')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    # optical flow parameters
    pars = [0.5, 0, 256, 4, 5, 1.1, 4]

    print(np.linalg.norm(data))
    print(theta)
    # ADMM solver
    for k in range(-10,10):
        print(k)
        with tc.SolverTomo(theta, ntheta, nz, n, pnz, (center+k)/pow(2, binning)) as tslv:
            ucg = tslv.cg_tomo_batch2(data, u, 64)
            dxchange.write_tiff_stack(
                            ucg.real,  '/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/cg'+'_'+str(ntheta)+'/rect'+'/r'+str((center+k)), overwrite=True)
        
