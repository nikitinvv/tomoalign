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
import tomopy 
def myplot(u, psi, flow, binning, iter):


    ids = np.argsort(flow[:,0]**2+flow[:,1]**2)[::-1]
    [ntheta, nz, n] = psi.shape

    plt.figure(figsize=(20, 14))
    for k in range(12):
        plt.subplot(3, 4, k+1)
        tit = 'y:',flow[ids[k],0],'x:',flow[ids[k],1]
        plt.title(tit)
        plt.imshow(psi[ids[k]], cmap='gray')

    if not os.path.exists('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/flow/'):
        os.makedirs('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/flow/')
    plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/flow/'+str(iter))
    plt.close()



def update_penalty(psi, h, h0, rho):
    # rho
    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho*(h-h0))**2
    if (r > 10*s):
        rho *= 2
    elif (s > 10*r):
        rho *= 0.5
    return rho

if __name__ == "__main__":
    binning = 2    
    prj = np.zeros([4000,512,612],dtype='float32')
    
    prj[:,:256,:] = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin2p1.npy').astype('float32')                                 
    #np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin0p1',a)        
    prj[:,256:,:] = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin2p2.npy').astype('float32')                                     
    #np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin0p2',a)        
    #exit()
    #prj=prj[500:-500,:,:]
    prj[np.isnan(prj)]=0
    # prj = tomopy.downsample(prj, level=binning)
    # prj = tomopy.downsample(prj, level=binning, axis=1)
    # np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin2p2',prj)        
    # #np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin2p2',prj[:,256:])        
    # exit()
    
    theta = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/theta.npy').astype('float32')
    # data
    data = prj.copy()
    [ntheta, nz, n] = data.shape  # object size n x,y
   
    center = 984
    niter = 64  # tomography iterations
    pnz = 128  # number of slice partitions for simultaneous processing in tomography    
    ptheta = 100
    # initial guess
    u = np.zeros([nz, n, n], dtype='float32')
    psi = data.copy()
    lamd = np.zeros([ntheta, nz, n], dtype='float32')    
    flow = np.zeros([ntheta, 2], dtype='float32')
    # ADMM solver
    print(np.linalg.norm(data))
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning),1) as tslv:
        # ucg = tslv.cg_tomo_batch(data, u, 32,dbg=True)
        # dxchange.write_tiff_stack(
        #                 ucg,  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cg/rect'+'/r', overwrite=True)
        # exit()
        with dc.SolverDeform(ntheta, nz, n, ptheta) as dslv:
            rho = 0.5
            h0 = psi.copy()
            for k in range(niter):
                # registration
                tic()
                if(k>1):
                    flow= dslv.registration_shift_batch(data, psi, 10)
                t1=toc()
                tic()
                # deformation subproblem
                psi,flow = dslv.cg_shift_batch(data, psi, flow, 4,
                                    tslv.fwd_tomo_batch(u)+lamd/rho, rho,dbg=False)
                #psi=data
                t2=toc()
                tic()
                u = tslv.cg_tomo_batch(psi-lamd/rho, u, 4)                
                t3=toc()
                h = tslv.fwd_tomo_batch(u)
                # lambda update
                lamd = lamd+rho*(h-psi)
                
                # checking intermediate results
                myplot(u, psi, flow, binning, k)
                if(np.mod(k, 1) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_shift_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = 0.5*np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, np.linalg.norm(flow), rho, lagr,t1,t2,t3)
                    dxchange.write_tiff_stack(
                        u,  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    
                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h.copy()
                
