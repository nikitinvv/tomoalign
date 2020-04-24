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

  
    ndsets = np.int(sys.argv[1])
    prj = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/battery/prj435bin2.npy')[0:ndsets*200].astype('float32')                                 
    theta = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/theta435.npy')[0:ndsets*200].astype('float32')
    print(prj.shape)
    #ext()
    # data
    data = prj[:,128:,50:-50].copy()-0.015
    #data[:,:,:32]=0
    #data[:,:,-32:]=0
    [ntheta, nz, n] = data.shape  # object size n x,y
    binning = 2    
    center = 1242-200
    niter = 256  # tomography iterations
    pnz = 64  # number of slice partitions for simultaneous processing in tomography    
    ptheta = 200
    # initial guess
    u = np.zeros([nz, n, n], dtype='float32')
    psi = 0*data.copy()
    
    lamd = np.zeros([ntheta, nz, n], dtype='float32')
    
    flow = np.zeros([ntheta, 2], dtype='float32')
    
    print(np.linalg.norm(data))
    # ADMM solver
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning),1) as tslv:
       # ucg = tslv.cg_tomo_batch(data, u, 32,dbg=False)
        #print(np.linalg.norm(tslv.fwd_tomo_batch(ucg)-data)**2)
      # # exit()
       # dxchange.write_tiff_stack(
            #            ucg.real,  'cg'+'_'+str(ntheta)+'/rect'+'/r', overwrite=True)
        with dc.SolverDeform(ntheta, nz, n, ptheta) as dslv:
            rho = 32
            h0 = psi.copy()
            for k in range(niter):
                # registration
                tic()
                if(k>1):
                    flow = dslv.registration_shift_batch(data, psi, 100)
                t1=toc()
                tic()
                # deformation subproblem
                psi,flow = dslv.cg_shift_batch(data, psi, flow, 4,
                                     tslv.fwd_tomo_batch(u)+lamd/rho, rho/2,dbg=False)
                #Tpsi = dslv.apply_shift_batch(psi, flow)
                
                #exit()
                t2=toc()
                tic()
                u = tslv.cg_tomo_batch(psi-lamd/rho, u, 4)                
                t3=toc()
                h = tslv.fwd_tomo_batch(u)
                # lambda update
                lamd = lamd+rho*(h-psi)
                
                # checking intermediate results
               # myplot(u, psi, flow, binning)
                if(np.mod(k, 1) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_shift_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho/2*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, np.linalg.norm(flow), rho, lagr,t1,t2,t3)
                    dxchange.write_tiff_stack(
                        u,  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/battery'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)


                    # print(flow[295])
                    # dxchange.write_tiff(
                    #     Tpsi[295]-data[295], '/data/staff/tomograms/vviknik/tomoalign_vincent_data/battery'+str(binning)+'_'+str(ntheta)+'/psir'+str(k),  overwrite=True)
                    # dxchange.write_tiff(
                    #     psi[295]-data[295], '/data/staff/tomograms/vviknik/tomoalign_vincent_data/battery1'+str(binning)+'_'+str(ntheta)+'/psir'+str(k),  overwrite=True)
                    
                # Updates
                #rho = update_penalty(psi, h, h0, rho)
                h0 = h.copy()
                
