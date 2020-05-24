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
import skimage.feature
matplotlib.use('Agg')
centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_721prj_180deg_1s_170': 1211,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167': 1224,
}
ngpus = 4
def apply_shift(psi, p):
    """Apply shift for all projections."""
    [nz,n] = psi.shape
    tmp = np.zeros([2*nz, 2*n], dtype='float32')
    tmp[nz//2:3*nz//2, n//2:3*n//2] = psi
    [x,y] = np.meshgrid(np.fft.rfftfreq(2*n),np.fft.fftfreq(2*nz))
    shift = np.exp(-2*np.pi*1j*(x*p[1]+y*p[0]))
    res0 = np.fft.irfft2(shift*np.fft.rfft2(tmp))
    res = res0[nz//2:3*nz//2, n//2:3*n//2]
    return res
        
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
    
    niter = 84
    binning=2
    data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(name+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
        [ntheta, nz, n] = data.shape  # object size n x,y
    [ntheta, nz, n] = data.shape  # object size n x,y
    data[np.isnan(data)]=0 
    data-=np.mean(data)
    mmin,mmax = find_min_max(data[:nth])
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    
    # pad data    
    ne = 3456//pow(2,binning)    
    center = centers[sys.argv[3]]+(ne//2-n//2)*pow(2,binning)        
    pnz = 8*pow(2,binning)  # number of slice partitions for simultaneous processing in tomography
    ptheta = 60

    pars = [0.5,0, ne, 32, 1, 1.1, 4]
    
    with dc.SolverDeform(nth, nz, n, 30) as dslv:
        for j in range(1,ndsets):
            flow = dslv.registration_flow_batch(
                        data[nth*j:nth*(j+1)], data[:nth], mmin, mmax, flow.copy(), pars, 20) 
            print('b',np.linalg.norm(data[nth*j:nth*(j+1)]-data[:nth]))
            data[nth*j:nth*(j+1)] = dslv.apply_flow_gpu_batch(data[nth*j:nth*(j+1)], flow)               
            print('a',np.linalg.norm(data[nth*j:nth*(j+1)]-data[:nth]))
    u = np.zeros([nz, ne, ne], dtype='float32')
    psi = data.copy()    
    
    with tc.SolverTomo(theta, ntheta, nz, ne, pnz, center/pow(2, binning), ngpus) as tslv:
        u = tslv.cg_tomo_batch(pad(psi,ne,n), u, niter)     
        dxchange.write_tiff_stack(
                        u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  name+'/alignedcg_'+str(binning)+'_'+str(ntheta)+'/rect''/r', overwrite=True)
        