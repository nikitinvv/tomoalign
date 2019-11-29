import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt


def myplot(u, psi, flow):
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
    plt.imshow(u[nz//2].real)
    plt.subplot(3, 4, 10)
    plt.imshow(u[nz//2+nz//8].real)

    plt.subplot(3, 4, 11)
    plt.imshow(u[:, n//2].real)

    plt.subplot(3, 4, 12)
    plt.imshow(u[:, :, n//2].real)
    if not os.path.exists('tmp'+'_'+str(ntheta)+'/'):
        os.makedirs('tmp'+'_'+str(ntheta)+'/')
    plt.savefig('tmp'+'_'+str(ntheta)+'/flow'+str(k))
    plt.close()

# Update penalty for ADMM


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

    prj = np.load('prj1.npy').astype('complex64')                                 
    theta = np.load('theta1.npy').astype('float32')

    # data
    data = prj[:,512-256:512+256,256+36:-256-36].copy()
    
    [ntheta, nz, n] = data.shape  # object size n x,y
    data[np.where(np.isnan(data))]=0
    print(data.shape)
    center = 1251-512-72
    binning = 1

    niter = 256  # tomography iterations
    pnz = 64  # number of slice partitions for simultaneous processing in tomography    

    # initial guess
    u = np.zeros([nz, n, n], dtype='complex64')
    psi = data.copy()
    lamd = np.zeros([ntheta, nz, n], dtype='complex64')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    # optical flow parameters
    pars = [0.5, 1, 512, 4, 5, 1.1, 0]

    print(np.linalg.norm(data))
    # ADMM solver
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning)) as tslv:
        # ucg = tslv.cg_tomo_batch2(data, u, 8)
        # dxchange.write_tiff_stack(
                        # ucg.real,  'cg'+'_'+str(ntheta)+'/rect'+'/r', overwrite=True)
        with dc.SolverDeform(ntheta, nz, n) as dslv:
            rho = 0.5
            h0 = psi
            for k in range(niter):
                # registration
                flow = dslv.registration_batch(psi, data, flow, pars)
                
                # deformation subproblem
                psi = dslv.cg_deform(data, psi, flow, 4,
                                     tslv.fwd_tomo_batch(u)+lamd/rho, rho)
                # tomo subproblem
                u = tslv.cg_tomo_batch2(psi-lamd/rho, u, 4)
                h = tslv.fwd_tomo_batch(u)
                # lambda update
                lamd = lamd+rho*(h-psi)

                # checking intermediate results
                myplot(u, psi, flow)
                if(np.mod(k, 4) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, pars[2], np.linalg.norm(flow), rho, lagr)
                    dxchange.write_tiff_stack(
                        u.real,  'tmp'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi.real, 'tmp'+str(binning)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                pars[2] -= 2
