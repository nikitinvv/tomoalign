import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from timing import tic,toc
import gc
import scipy.ndimage as ndimage
matplotlib.use('Agg')
matplotlib.use('Agg')
centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/zp/Kenan_ZP_9100eV_interlaced_30deg_2.5s_061': 512,
}
ngpus = 4

        
def find_min_max(data):
    # s = np.std(data,axis=(1,2))    
    # m = np.mean(data,axis=(1,2))
    # mmin = m-2*s
    # mmax = m+2*s
    mmin = np.zeros(data.shape[0],dtype='float32')
    mmax = np.zeros(data.shape[0],dtype='float32')
    
    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:],1000)
        stend = np.where(h>np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]        
        mmin[k] = e[st]
        mmax[k] = e[end+1]
     
    return mmin,mmax

def pad(data,ne,n):
    datae = np.zeros([data.shape[0],nz,ne],dtype='float32')
    datae[:,:,ne//2-n//2:ne//2+n//2]=data
    datae[:,:,:ne//2-n//2]=datae[:,:,ne//2-n//2:ne//2-n//2+1]
    datae[:,:,ne//2+n//2:]=datae[:,:,ne//2+n//2-1:ne//2+n//2]
    return datae

def unpad(data,ne,n):
    return data[:,:,ne//2-n//2:ne//2+n//2]


if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]   
    
    niter = 32
    binning=1
    data = np.zeros([ndsets*nth,2048//2//pow(2,binning),2048//2//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(name+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
        [ntheta, nz, n] = data.shape  # object size n x,y
    [ntheta, nz, n] = data.shape  # object size n x,y
    dxchange.write_tiff_stack(data,name+'/datasave',overwrite=True)
    data[np.isnan(data)]=0            
    data-=np.mean(data)
    mmin,mmax = find_min_max(data)
    # pad data    
    ne = 3072//2//pow(2,binning)    
    #ne=n
    center = centers[sys.argv[3]]+(ne//2-n//2)*pow(2,binning)        
    pnz = 8*pow(2,binning)  # number of slice partitions for simultaneous processing in tomography
    ptheta = 60

       
    u = np.zeros([nz, ne, ne], dtype='float32')
    psi = data.copy()
    lamd = np.zeros([ntheta, nz, n], dtype='float32')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
       
    with tc.SolverTomo(theta, ntheta, nz, ne, pnz, center/pow(2, binning), ngpus) as tslv:
        u = tslv.cg_tomo_batch(pad(psi,ne,n), u, niter)     
        dxchange.write_tiff_stack(
                        u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  name+'/cg_'+'_'+str(ntheta)+'/rect''/r', overwrite=True)
        