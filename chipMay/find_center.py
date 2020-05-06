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
#import tomopy 
# matplotlib.use('Agg')
# def myplot(u, psi, flow, binning):
#     ids = np.argsort(np.linalg.norm(flow,axis=(1,2,3)))[::-1]
#     [ntheta, nz, n] = psi.shape

#     plt.figure(figsize=(20, 14))
#     for k in range(4):
#         plt.subplot(2, 4, k+1)
#         tit = 'y:',flow[ids[k],0],'x:',flow[ids[k],1]
#         plt.title(tit)
#         plt.imshow(psi[ids[k]], cmap='gray')
#name='/local/data/viktor/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167'
# prj=np.load('/local/data/viktor/brain_rec/cprjbin0p2.npy')
# print(prj.shape)
# prj = tomopy.downsample(prj, level=1)
# prj = tomopy.downsample(prj, level=1, axis=1)
# print(np.linalg.norm(prj[-1]))
# exit()
# np.save('/local/data/viktor/brain_rec/cprjbin1p2',prj)        
# prj = tomopy.downsample(prj, level=1)
# prj = tomopy.downsample(prj, level=1, axis=1)
# np.save('/local/data/viktor/brain_rec/cprjbin2p2',prj)        
def find_min_max(data):
    s = np.std(data,axis=(1,2))    
    m = np.mean(data,axis=(1,2))
    mmin = m-2*s
    mmax = m+2*s
    return mmin,mmax

if __name__ == "__main__":
    binning = 2
    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]
    data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(name+'_bin2'+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
    
    data=data[:,data.shape[1]//2:data.shape[1]//2+1]
    data[np.isnan(data)]=0    
    center = np.float(sys.argv[4])
    print('shape',data.shape,'center',center//pow(2,binning))
    
    [ntheta, nz, n] = data.shape  # object size n x,y
    data-=np.mean(data)
    
    # exit()
    niter = 32  # tomography iterations
    pnz = 1  # number of slice partitions for simultaneous processing in tomography
    ngpus = 1
    # initial guess
    u = np.zeros([nz, n, n], dtype='float32')
    psi = data.copy()

    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning), ngpus) as tslv:
        ucg = tslv.cg_tomo_batch(data, u, niter)
        dxchange.write_tiff(ucg[0],  name+'/cg'+'_'+str(ntheta)+'/rect'+str(center), overwrite=True)                    