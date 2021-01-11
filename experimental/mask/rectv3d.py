import tomorectv3d
import dxchange
import numpy as np
import sys

if __name__ == "__main__":
    fname = sys.argv[1]   
    ndsets = int(sys.argv[2])
    nth = int(sys.argv[3])
    binning = int(sys.argv[4])    
    center = float(sys.argv[5])
    lambda0 = float(sys.argv[6])
    st = int(sys.argv[7])
    end = int(sys.argv[8])
    
    data = np.zeros([ndsets*nth,((end-st))//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin'+str(binning)+str(k)+'.npy')[:,st:end].astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta'+str(k)+'.npy').astype('float32')
    data[np.isnan(data)]=0           
    data = data.swapaxes(0,1)
    [Nz,Ntheta,N] = data.shape
    print('Data shape', [Nz,Ntheta,N])    
    
    Nzp = 8  # number of slices for simultaneous processing by 1 gpu
    ngpus = 4  # number of gpus to process the data (index 0,1,2,.. are taken)
    niter = 32  # number of iterations in the Chambolle-Pock algorithm
    method = 0
    # read object
    with tomorectv3d.Solver(N, Ntheta, Nz, Nzp,  method, ngpus, center, lambda0) as cl:
        # set angles
        cl.settheta(theta)
        # reconstruction with 3d tv
        res = np.zeros([Nz, N, N], dtype='float32', order='C')
        data = np.ascontiguousarray(data)
        cl.itertvR(res, data, niter)
        dxchange.write_tiff_stack(res, fname+'_rec/r.tiff', overwrite=True)