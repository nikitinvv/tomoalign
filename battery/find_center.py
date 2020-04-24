import dxchange
import numpy as np
import tomocg as tc
import scipy as sp
import sys
import os



if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    data = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/battery/prj435bin2.npy')[0:ndsets*200].astype('float32')                                 
    print(data.shape)
    #dxchange.write_tiff_stack(data,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/battery/data/d')
    theta = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/theta435.npy')[0:ndsets*200].astype('float32')
    
    data = data[:,200:201]-0.015
    #print(data.shape)
    # data
    data[np.isnan(data)]=0
    [ntheta, nz, n] = data.shape  # object size n x,y
    
    center = data.shape[2]//2
    center = 1242
    binning = 2
    niter = 64  # tomography iterations
    pnz = 1  # number of slice partitions for simultaneous processing in tomography    

    # initial guess
    u = np.zeros([nz, n, n], dtype='float32')
    print(np.linalg.norm(data))
    #print(theta)
    # ADMM solver
    center = center+np.int(sys.argv[2])
    print(center)
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning),1) as tslv:
        ucg = tslv.cg_tomo_batch(data, u, niter)
        dxchange.write_tiff_stack(
                        ucg,  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/battery/cg'+'_'+str(ntheta)+'/rect'+'/r'+str(center), overwrite=False)
        
