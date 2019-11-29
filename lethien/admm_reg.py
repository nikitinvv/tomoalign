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
    data = prj[:,512-8:512+8,256+36:-256-36].copy()
    
    [ntheta, nz, n] = data.shape  # object size n x,y
    data[np.where(np.isnan(data))]=0
    print(data.shape)
    center = 1251-512-72
    binning = 1

    niter = 256  # tomography iterations
    pnz = 16  # number of slice partitions for simultaneous processing in tomography    
    alpha = 1e-3
    # initial guess
    u = np.zeros([nz, n, n], dtype='complex64')
    psi1 = data.copy()
    psi2 = np.zeros([ntheta, nz, n], dtype='complex64')
    
    lamd1 = np.zeros([ntheta, nz, n], dtype='complex64')
    lamd2 = np.zeros([3, nz, n, n], dtype='complex64')
    
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
            rho1 = 0.5
            rho2 = 0.5
            
            h01 = psi1
            h02 = psi2
            
            for k in range(niter):
                # registration
                flow = dslv.registration_batch(psi1, data, flow, pars)
                
                # deformation subproblem
                psi1 = dslv.cg_deform(data, psi1, flow, 4,
                                     tslv.fwd_tomo_batch(u)+lamd1/rho1, rho1)
                psi2 = tslv.solve_reg(u,lamd2,rho2,alpha)                     
                # tomo subproblem
                u = tslv.cg_tomo_batch_ext(psi1-lamd1/rho1, psi2-lamd2/rho2, u, rho2/rho1, 4)
                h1 = tslv.fwd_tomo_batch(u)
                h2 = tslv.fwd_reg(u)
                # lambda update
                lamd1 = lamd1+rho1*(h1-psi1)
                lamd2 = lamd2+rho2*(h2-psi2)

                # checking intermediate results
                myplot(u, psi1, flow)
                if(np.mod(k, 4) == 0):  # check Lagrangian
                    Tpsi = dslv.apply_flow_batch(psi1, flow)
                    lagr = np.zeros(7)
                    lagr[0] = np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd1)*(h1-psi1)))
                    lagr[2] = rho1*np.linalg.norm(h1-psi1)**2
                    lagr[3] = alpha*np.sum(np.sqrt(np.real(np.sum(psi2*np.conj(psi2), 0))))
                    lagr[4] = np.sum(np.real(np.conj(lamd2*(h2-psi2))))
                    lagr[5] = rho2*np.linalg.norm(h2-psi2)**2
                    lagr[6] = np.sum(lagr[0:5])
                    print(k, pars[2], np.linalg.norm(flow), rho1, rho2, lagr)
                    dxchange.write_tiff_stack(
                        u.real,  'tmp'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi1.real, 'tmp'+str(binning)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho1 = update_penalty(psi1, h1, h01, rho1)
                rho2 = update_penalty(psi2, h2, h02, rho2)
                h01 = h1
                h02 = h2
                pars[2] -= 2
