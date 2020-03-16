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

<<<<<<< HEAD
from datetime import datetime


def myplot(u, psi, flow, binning, alpha):
=======
def myplot(u, psi, flow, binning):
>>>>>>> 318412382bacc9b7d9c376db65ab2be998ddf4e6
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
    if not os.path.exists('/data/staff/tomograms/viknik/tomoalign_vincent_data/chip/flow/'):
        os.makedirs('/data/staff/tomograms/viknik/tomoalign_vincent_data/chip/flow/')
    plt.savefig('/data/staff/tomograms/viknik/tomoalign_vincent_data/chip/flow/flow'+str(k))
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
<<<<<<< HEAD
    alpha = np.float32(sys.argv[2])
    prj = np.load('prjbin1.npy')[0:ndsets*200].astype('complex64')[:,32:-32]                                 
    theta = np.load('theta.npy')[0:ndsets*200].astype('float32')
=======
    prj = np.load('/data/staff/tomograms/viknik/tomoalign_vincent_data/prjbin1.npy')[0:ndsets*200].astype('complex64')                                 
    theta = np.load('/data/staff/tomograms/viknik/tomoalign_vincent_data/theta.npy')[0:ndsets*200].astype('float32')
>>>>>>> 318412382bacc9b7d9c376db65ab2be998ddf4e6

    # data
    data = prj.copy()
    [ntheta, nz, n] = data.shape  # object size n x,y
    
    center = 1168-456
<<<<<<< HEAD
    binning = 1   
    niter = 256  # tomography iterations
    pnz = 64  # number of slice partitions for simultaneous processing in tomography    
=======
    binning = 1
    niter = 1  # tomography iterations
    pnz = 32  # number of slice partitions for simultaneous processing in tomography    
>>>>>>> 318412382bacc9b7d9c376db65ab2be998ddf4e6

    # initial guess
    u = np.zeros([nz, n, n], dtype='complex64')
    psi1 = data.copy()*0
    psi2 = np.zeros([3,nz, n, n], dtype='complex64')
    
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64')
    lamd2 = np.zeros([3, nz, n, n], dtype='complex64')
    
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    # optical flow parameters
<<<<<<< HEAD
    pars=[1, 0.5, True, 256, 4, 5, 1.1, 4]
=======
    pars = [0.5, 0, 256, 4, 5, 1.1, 4]

    print(np.linalg.norm(data))
>>>>>>> 318412382bacc9b7d9c376db65ab2be998ddf4e6
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
<<<<<<< HEAD
                if(k>0):
                    print(datetime.now().strftime("%H:%M:%S"),'reg')
                    flow = dslv.registration_flow_batch(psi1, data, flow, pars)                
                # deformation subproblem
                print(datetime.now().strftime("%H:%M:%S"),'def')
                psi1 = dslv.cg_deform(data, psi1, flow, 4,
                                     tslv.fwd_tomo_batch(u)+lamd1/rho1, rho1)
                psi2 = tslv.solve_reg(u,lamd2,rho2,alpha)    
                # tomo subproblem
                print(datetime.now().strftime("%H:%M:%S"),'tomo')
                u = tslv.cg_tomo_batch_ext2(data, u, 4, rho2/rho1, psi2-lamd2/rho2)
                print(datetime.now().strftime("%H:%M:%S"),'other')
                h1 = tslv.fwd_tomo_batch(u)
                h2 = tslv.fwd_reg(u)
                # lambda update
                lamd1 = lamd1+rho1*(h1-psi1)
                lamd2 = lamd2+rho2*(h2-psi2)
                # checking intermediate results
                myplot(u, psi1, flow, binning, alpha)
                if(np.mod(k, 1) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi1, flow)
                    lagr = np.zeros(7)
=======
                tic()
                flow = dslv.registration_flow_batch(psi, data, flow.copy(), pars,64)
                print(toc())
                tic()
                # deformation subproblem
                psi = dslv.cg_deform(data, psi, flow, 2,
                                     tslv.fwd_tomo_batch(u)+lamd/rho, rho,64)
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
>>>>>>> 318412382bacc9b7d9c376db65ab2be998ddf4e6
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, pars[2], np.linalg.norm(flow), rho, lagr)
                    dxchange.write_tiff_stack(
<<<<<<< HEAD
                        u.real,  'tmp'+str(binning)+str(alpha)+'_'+str(ntheta)+'/rect'+str(k)+'/r',overwrite=True)
                    dxchange.write_tiff_stack(
                        psi1.real, 'tmp2'+str(binning)+str(alpha)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho1 = update_penalty(psi1, h1, h01, rho1)
                rho2 = update_penalty(psi2, h2, h02, rho2)
                h01 = h1.copy()
                h02 = h2.copy()
                pars[3] -= 1
=======
                        u.real,  '/data/staff/tomograms/viknik/tomoalign_vincent_data/chip'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi.real, '/data/staff/tomograms/viknik/tomoalign_vincent_data/chip'+str(binning)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                pars[2] -= 1
>>>>>>> 318412382bacc9b7d9c376db65ab2be998ddf4e6
