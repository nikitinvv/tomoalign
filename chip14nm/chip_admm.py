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
matplotlib.use('Agg')

def myplot(u, psi, flow, binning):
    [ntheta, nz, n] = psi.shape

    plt.figure(figsize=(20, 14))
    plt.subplot(3, 4, 1)
    plt.imshow(psi[ntheta//4].real, cmap='gray')

    plt.subplot(3, 4, 2)
    plt.imshow(psi[ntheta//2].real, cmap='gray')
    plt.subplot(3, 4, 3)
    plt.imshow(psi[3*ntheta//4].real, cmap='gray')

    plt.subplot(3, 4, 4)
    plt.imshow(psi[-1].real, cmap='gray')

    plt.subplot(3, 4, 5)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//4]), cmap='gray')

    plt.subplot(3, 4, 6)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//2]), cmap='gray')

    plt.subplot(3, 4, 7)
    plt.imshow(dc.flowvis.flow_to_color(flow[3*ntheta//4]), cmap='gray')
    plt.subplot(3, 4, 8)
    plt.imshow(dc.flowvis.flow_to_color(flow[-1]), cmap='gray')

    plt.subplot(3, 4, 9)
    plt.imshow(u[nz//2].real,cmap='gray')
    plt.subplot(3, 4, 10)
    plt.imshow(u[nz//2+nz//8].real,cmap='gray')

    plt.subplot(3, 4, 11)
    plt.imshow(u[:, n//2].real,cmap='gray')

    plt.subplot(3, 4, 12)
    plt.imshow(u[:, :, n//2].real,cmap='gray')
    if not os.path.exists('/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/flow'+str(binning)+'_'+str(ntheta)+'/'):
        os.makedirs('/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/flow'+str(binning)+'_'+str(ntheta)+'/')
    plt.savefig('/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/flow'+str(binning)+'_'+str(ntheta)+'/'+str(k))
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

    ndsets = np.int(sys.argv[1])
    prj = np.load('/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/prjbin2.npy')[0:ndsets*200].astype('complex64')                                 
    theta = np.load('/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/theta.npy')[0:ndsets*200].astype('float32')

    # data
    binning = 2
    data = prj[:,256//pow(2,binning):-384//pow(2,binning)].copy()
    data[np.isnan(data)]=0
    [ntheta, nz, n] = data.shape  # object size n x,y
    
    center = 1250
    
    niter = 256  # tomography iterations
    pnz = 32  # number of slice partitions for simultaneous processing in tomography    

    # initial guess
    u = np.zeros([nz, n, n], dtype='complex64')
    psi = data.copy()
    lamd = np.zeros([ntheta, nz, n], dtype='complex64')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    # optical flow parameters
    pars = [0.5, 0, 256, 4, 5, 1.1, 4]

    print(np.linalg.norm(data))
    # ADMM solver
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning)) as tslv:
        #ucg = tslv.cg_tomo_batch2(data, u, 8)
        #dxchange.write_tiff_stack(
                        #ucg.real,  'cg'+'_'+str(ntheta)+'/rect'+'/r', overwrite=True)
        with dc.SolverDeform(ntheta, nz, n) as dslv:
            rho = 0.5
            h0 = psi
            for k in range(niter):
                # registration
                tic()
                flow = dslv.registration_flow_batch(psi, data, flow.copy(), pars,nproc=14)
                print(toc())
                tic()
                # deformation subproblem
                psi = dslv.cg_deform(data, psi, flow, 2,
                                     tslv.fwd_tomo_batch(u)+lamd/rho, rho,nproc=14)
                print(toc())
                # tomo subproblem                
                tic()
                u = tslv.cg_tomo_batch(psi-lamd/rho, u, 4)
                print(toc())
                h = tslv.fwd_tomo_batch(u)
                # lambda update
                lamd = lamd+rho*(h-psi)

                # checking intermediate results
                myplot(u, psi, flow, binning)
                if(np.mod(k, 4) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, pars[2], np.linalg.norm(flow), rho, lagr)
                    dxchange.write_tiff_stack(
                        u.real,  '/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/chip'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi.real, '/data/staff/tomograms/viknik/tomoalign_vincent_data/14nmZP/chip'+str(binning)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                pars[2] -= 1
