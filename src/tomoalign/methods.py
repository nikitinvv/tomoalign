"""Methods for alignment projections and reconstruction"""

import numpy as np

from .solver_tomo import SolverTomo
from .solver_deform import SolverDeform
from .utils import *
import dxchange


def prealign(data, pprot):
    """prealign projections by optical flow according to adjacent interlaced angles"""

    [ntheta, nz, n] = data.shape
    mmin, mmax = find_min_max(data)
    # parameters for non-dense flow in Farneback's algorithm,
    # resulting flow is constant, i.e. equivalent to a shift
    pars = [0.5, 1, 2*n, 4, 5, 1.1, 4]
    res = data.copy()
    for k in range(1, ntheta//pprot):
        with SolverDeform(pprot, nz, n, 16, 1) as dslv:
            flow = dslv.registration_flow_batch(
                data[k*pprot:(k+1)*pprot], data[0*pprot:(0+1)*pprot], mmin[:pprot], mmax[:pprot], None, pars)
            res[k*pprot:(k+1)*pprot] = dslv.apply_flow_gpu_batch(data[k *
                                                                      pprot:(k+1)*pprot], flow)
    return res


def pcg(data, theta, pprot, pnz, center, ngpus, niter):
    """Reconstruct with the _prealigned CG (pCG)"""

    [ntheta, nz, n] = data.shape
    ne = 3*n//2  # padded size
    u = np.zeros([nz, ne, ne], dtype='float32')
    # tomographic solver on GPU
    with SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        u = unpadobject(tslv.cg_tomo_batch(
            paddata(prealign(data, pprot), ne, n), u, niter), ne, n)
    return u


def admm_of(data, theta, pnz, ptheta, center, stepwin, ngpus, niter, titer):
    """Reconstruct with the optical flow method (OF)"""

    [ntheta, nz, n] = data.shape
    ne = 3*n//2  # padded size
    # tomographic solver on GPU
    with SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        # alignment solver on GPU
        with SolverDeform(ntheta, nz, n, ptheta, ngpus) as dslv:
            # find min,max values accoding to histogram
            mmin, mmax = find_min_max(data)

            # initial guess and coordinating variables
            u = np.zeros([nz, ne, ne], dtype='float32')
            psi = data.copy()
            h0 = psi.copy()
            lamd = np.zeros([ntheta, nz, n], dtype='float32')
            flow = np.zeros([ntheta, nz, n, 2], dtype='float32')

            # optical flow parameters (see openCV function for Farneback's algorithm)
            pars = [0.5, 1, n, titer, 5, 1.1, 4]
            rho = 0.5  # weighting factor in ADMM
            lagr = np.zeros([niter, 4],dtype='float32')

            t = np.zeros([4])
            for k in range(niter):

                # 1. Solve the alignment sub-problem
                # register flow
                tic()
                flow = dslv.registration_flow_batch(
                    psi, data, mmin, mmax, flow, pars)
                t[0] = toc()
                # unwarping
                tic()
                psi = dslv.cg_deform_gpu_batch(data, psi, flow, titer, unpaddata(
                    tslv.fwd_tomo_batch(u), ne, n)+lamd/rho, rho)
                t[1] = toc()
                # 2. Solve the tomography sub-problen
                tic()
                u = tslv.cg_tomo_batch(paddata(psi-lamd/rho, ne, n), u, 4)
                t[2] = toc()

                # compute forward tomography operator for further updates of rho and lambda
                tic()
                h = unpaddata(tslv.fwd_tomo_batch(u), ne, n)
                t[3] = toc()
                # 3. dual update
                lamd = lamd+rho*(h-psi)
                print(*t, np.sum(t))

                if(np.mod(k, 1) == 0):  # check Lagrangian, save current iteration results
                    Tpsi = dslv.apply_flow_gpu_batch(psi, flow)
                    lagr[k, 0] = 0.5*np.linalg.norm(Tpsi-data)**2
                    lagr[k, 1] = np.sum(lamd*(h-psi))
                    lagr[k, 2] = 0.5*rho*np.linalg.norm(h-psi)**2
                    lagr[k, 2] = 0.5*rho*np.linalg.norm(h-psi)**2
                    lagr[k, 3] = np.sum(lagr[k,0:3])
                    print("iter %d, wsize %d, rho %.2f, Lagrangian %.4e %.4e %.4e Total %.4e " % (
                        k, pars[2], rho, *lagr[k]))
                    # save object
                    dxchange.write_tiff(unpadobject(u, ne, n),  'data/of_recon/recon/iter'+str(k), overwrite=True)
                    # save flow figure
                    dslv.flowplot(u, psi, flow, 'data/of_recon/flow/iter'+str(k))

                # update rho
                rho = _update_penalty(psi, h, h0, rho)
                h0 = h

                if(pars[2] > 12):  # limit optical flow window size
                    pars[2] -= stepwin
    return u, lagr


def admm_of_reg(data, theta, pnz, ptheta, center, stepwin, alpha, ngpus, niter, titer):
    """Reconstruct with the optical flow method and regularization (OFTV)"""

    [ntheta, nz, n] = data.shape
    ne = 3*n//2  # padded size
    # tomographic solver on GPU
    lagr = np.zeros([niter, 6],dtype='float32')

    with SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        # alignment solver on GPU
        with SolverDeform(ntheta, nz, n, ptheta, ngpus) as dslv:
            # find min,max values accoding to histogram
            mmin, mmax = find_min_max(data)

            # initial guess and coordinating variables
            u = np.zeros([nz, ne, ne], dtype='float32')
            psi1 = data.copy()
            psi2 = np.zeros([3, nz, ne, ne], dtype='float32')

            h01 = psi1.copy()
            h02 = psi2.copy()
            lamd1 = np.zeros([ntheta, nz, n], dtype='float32')
            lamd2 = np.zeros([3, nz, ne, ne], dtype='float32')
            flow = np.zeros([ntheta, nz, n, 2], dtype='float32')

            # optical flow parameters (see openCV function for Farneback's algorithm)
            pars = [0.5, 1, n, titer, 5, 1.1, 4]
            rho1 = 0.5  # weighting factor in ADMM w.r.t. tomo sub-problem
            rho2 = 0.5  # weighting factor in ADMM w.r.t. deform sub-problem

            for k in range(niter):

                # 1. Solve the alignment sub-problem
                # register flow
                flow = dslv.registration_flow_batch(
                    psi1, data, mmin, mmax, flow, pars)
                # unwarping
                psi1 = dslv.cg_deform_gpu_batch(data, psi1, flow, titer, unpaddata(
                    tslv.fwd_tomo_batch(u), ne, n)+lamd1/rho1, rho1)

                # 2. Solve the regularization sub-problen
                psi2 = tslv.solve_reg(u, lamd2, rho2, alpha)

                # 3. Solve the tomography sub-problen
                u = tslv.cg_tomo_reg_batch(
                    paddata(psi1-lamd1/rho1, ne, n), u, titer, rho2/rho1, psi2-lamd2/rho2)

                # compute forward operators for further updates of rho and lamd
                h1 = unpaddata(tslv.fwd_tomo_batch(u), ne, n)
                h2 = tslv.fwd_reg(u)

                # 4. dual update
                lamd1 = lamd1+rho1*(h1-psi1)
                lamd2 = lamd2+rho2*(h2-psi2)

                if(np.mod(k, 1) == 0):  # check Lagrangian, save current iteration results
                    Tpsi1 = dslv.apply_flow_gpu_batch(psi1, flow)
                    lagr[k, 0] = 0.5*np.linalg.norm(Tpsi1-data)**2
                    lagr[k, 1] = np.sum(lamd1*(h1-psi1))
                    lagr[k, 2] = 0.5*rho1*np.linalg.norm(h1-psi1)**2
                    lagr[k, 3] = 0.5*rho2*np.linalg.norm(h2-psi2)**2
                    lagr[k, 4] = np.sum(np.sqrt(np.sum(psi2**2, 0)))
                    lagr[k, 5] = np.sum(lagr[k,0:5])
                    print("iter %d, wsize %d, rho (%.2f,%.2f), Lagrangian terms %.4e %.4e %.4e %.4e %.4e Total %.4e " % (
                        k, pars[2], rho1, rho2, *lagr[k]))
                    # save object
                    dxchange.write_tiff(unpadobject(
                    u, ne, n),  'data/of_recon_reg/recon/iter'+str(k), overwrite=True)
                    # save flow figure
                    dslv.flowplot(u, psi1, flow, 'data/of_recon_reg/flow/iter'+str(k))

                # update rho
                rho1 = _update_penalty(psi1, h1, h01, rho1)
                rho2 = _update_penalty(psi2, h2, h02, rho2)

                h01 = h1
                h02 = h2

                if(pars[2] > 12):  # limit optical flow window size
                    pars[2] -= stepwin
    return u, lagr


def _update_penalty(psi, h, h0, rho):
    """Update the ADMM weighting penalty for faster convergence"""

    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho*(h-h0))**2
    if (r > 10*s):
        rho *= 2
    elif (s > 10*r):
        rho *= 0.5
    return rho
