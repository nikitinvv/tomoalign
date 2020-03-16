import cupy as cp
import dxchange
import numpy as np
import tomocg
import deformcg


def update_penalty(psi, h, h0, rho):
    """Update penalty Lagrangian factor rho for faster convergence"""
    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho*(h-h0))**2
    if (r > 10*s):
        rho *= 2
    elif (s > 10*r):
        rho *= 0.5
    return rho


def take_lagr(psi,h,data,lamd,shift,shift0,rho,k):
    """Compute Lagrangian, save intermediate results"""

    Tpsi = ds.apply_shift_batch(psi, shift)
    lagr = np.zeros(4)
    lagr[0] = np.linalg.norm(Tpsi-data)**2
    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
    lagr[2] = rho*np.linalg.norm(h-psi)**2
    lagr[3] = np.sum(lagr[0:3])
    print('iter:',k,'rho:', rho,'Lagrangian terms:',lagr)
    print('angle0',shift[0],'<->',shift0[0],'dif:',shift[0]-shift0[0])
    print('angle1',shift[-1],'<->',shift0[-1],'dif:',shift[-1]-shift0[-1])
    dxchange.write_tiff_stack(
        u.real,  'rec/admm/delta'+str(k)+'/r', overwrite=True)
    dxchange.write_tiff_stack(
        psi.real, 'rec/admm/psi'+str(k)+'/r',  overwrite=True)


if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 64  # number of angles (rotations)
    center = n/2  # rotation center
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles
    pnz = 128  # number of slice partitions for simultaneous processing in tomography
    # Load object
    beta = dxchange.read_tiff('data/beta-chip-128.tiff')
    delta = dxchange.read_tiff('data/delta-chip-128.tiff')
    u0 = delta+1j*beta
    print(u0.shape)
   
    with tomocg.SolverTomo(theta, ntheta, nz, n, pnz, center) as ts:
        with deformcg.SolverDeform(ntheta, nz, n) as ds:
            # generate data
            data0 = ts.fwd_tomo_batch(u0)
            # shift data
            shift0 = (np.random.random([ntheta, 2])-0.5)*4
            data = ds.apply_shift_batch(data0, shift0)


            # 1) Solution by CG
            print('Standard CG solver')
            u = np.zeros([nz,n,n],dtype='complex64')
            # solution by standard cg
            ucg = ts.cg_tomo_batch(data,u,64)
            
            dxchange.write_tiff(ucg.imag,  'rec/cg/betacg', overwrite=True)
            dxchange.write_tiff(ucg.real,  'rec/cg/deltacg', overwrite=True)


            # 2) Solution by ADMM with alignment
            print('ADMM solver with alignment')
            # Initial guess
            u = np.zeros([nz, n, n], dtype='complex64')
            psi = data#np.zeros([ntheta, nz, n], dtype='complex64')
            lamd = np.zeros([ntheta, nz, n], dtype='complex64')

            # ADMM
            rho = 0.5  # Lagrangian variable
            h0 = psi
            for k in range(64):
                # registration
                shift = ds.registration_shift_batch(data, psi, upsample_factor=1)# dynamically increase upsampling
                # explicit solution for the translation subproblem
                psi = (ds.apply_shift_batch(data, -shift) +
                       rho*ts.fwd_tomo_batch(u)+lamd)/(1+rho)
                # tomo subproblem
                u = ts.cg_tomo_batch(psi-lamd/rho, u, 4)  # 4 inner iterations
                # lambda update
                h = ts.fwd_tomo_batch(u)
                lamd = lamd+rho*(h-psi)

                if(np.mod(k, 4) == 0):# checking intermediate results
                    take_lagr(psi,h,data,lamd,shift,shift0,rho,k)   
                
                # Updates for faster convergence
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
