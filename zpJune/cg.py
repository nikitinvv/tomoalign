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
centers={
'/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090': 1160,
}
ngpus = 4

def pad(data,ne,n):
    datae = np.zeros([data.shape[0],nz,ne],dtype='float32')
    datae[:,:,ne//2-n//2:ne//2+n//2]=data
    datae[:,:,:ne//2-n//2]=datae[:,:,ne//2-n//2:ne//2-n//2+1]
    datae[:,:,ne//2+n//2:]=datae[:,:,ne//2+n//2-1:ne//2+n//2]
    return datae

def unpad(data,ne,n):
    return data[:,:,ne//2-n//2:ne//2+n//2]

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

def prealign(data):
    mmin,mmax = find_min_max(data)
    pars = [0.5,1, 2*n, 4, 5, 1.1, 4]
    for k in range(ndsets):
       with dc.SolverDeform(nth, nz, n, 20) as dslv:
           flow = dslv.registration_flow_batch(
                        data[k*nth:(k+1)*nth], data[0*nth:(0+1)*nth], mmin[:nth], mmax[:nth], None, pars, 20) 
           data[k*nth:(k+1)*nth] = dslv.apply_flow_gpu_batch(data[k*nth:(k+1)*nth], flow)
    return data

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]   
    
    binning = 2
    data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(name+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
    [ntheta, nz, n] = data.shape  # object size n x,y
    
    data[np.isnan(data)]=0            
    data-=np.mean(data)
    # pad data    
    ne = (2048+1024)//pow(2,binning)    
    #ne=n
    print(data.shape)
    data=prealign(data)
    center = centers[sys.argv[3]]+(ne//2-n//2)*pow(2,binning)        
    pnz = 8*pow(2,binning)  # number of slice partitions for simultaneous processing in tomography
    u = np.zeros([nz, ne, ne], dtype='float32')
    with tc.SolverTomo(theta, ntheta, nz, ne, pnz, center/pow(2, binning), ngpus) as tslv:
        ucg = tslv.cg_tomo_batch(pad(data,ne,n),u,32)
        dxchange.write_tiff_stack(
            ucg[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  name+'/cga_'+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
    