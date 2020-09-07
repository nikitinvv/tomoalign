"""Methods for alignment projections and reconstruction"""

import numpy as np

from .solver_tomo import SolverTomo
from .solver_deform import SolverDeform
from .utils import *
import dxchange
import sys
import scipy as sp


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
    ne = _take_psize(n)
    u = np.zeros([nz, ne, ne], dtype='float32')
    # tomographic solver on GPU
    with SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        u = unpadobject(tslv.cg_tomo_batch(
            paddata(prealign(data, pprot), ne, n), u, niter), ne, n)
    res = {'u': u}
    return res

def pinv(data, theta, pprot, pnz, center, ngpus, niter):
    """Reconstruct with the _prealigned CG (pCG)"""

    [ntheta, nz, n] = data.shape
    ne = _take_psize(n)
    u = np.zeros([nz, ne, ne], dtype='float32')
    # tomographic solver on GPU
    with SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        u = unpadobject(tslv.inv_tomo_batch(
            paddata(prealign(data, pprot), ne, n), u, niter), ne, n)
    res = {'u': u}
    return res

def admm_of(data, theta, pnz, ptheta, center, ngpus, niter, startwin, stepwin, res=None, fname='', titer=4):
    """Reconstruct with the optical flow method (OF)"""
    [ntheta, nz, n] = data.shape
    # tomographic solver on GPU
    ne = _take_psize(n)

    with SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        # alignment solver on GPU
        with SolverDeform(ntheta, nz, n, ptheta, ngpus) as dslv:
            # find min,max values accoding to histogram
            mmin, mmax = find_min_max(data)

            # initial guess and coordinating variables
            if(res == None):
                u = np.zeros([nz, ne, ne], dtype='float32')
                psi = data.copy()
                h0 = psi.copy()
                lamd = np.zeros([ntheta, nz, n], dtype='float32')
                flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
                res = {}
            else:
                u = res['u']
                psi = res['psi']
                h0 = res['h0']
                lamd = res['lamd']
                flow = res['flow']

            # optical flow parameters (see openCV function for Farneback's algorithm)
            pars = [0.5, 1, startwin, titer, 5, 1.1, 4]
            rho = 0.5  # weighting factor in ADMM
            lagr = np.zeros([niter, 4], dtype='float32')

            for k in range(niter):

                # 1. Solve the alignment sub-problem
                # register flow
                flow = dslv.registration_flow_batch(
                    psi, data, mmin, mmax, flow, pars)

                # unwarping
                psi = dslv.cg_deform_gpu_batch(data, psi, flow, titer, unpaddata(
                    tslv.fwd_tomo_batch(u), ne, n)+lamd/rho, rho)
                # 2. Solve the tomography sub-problen
                u = tslv.cg_tomo_batch(paddata(psi-lamd/rho, ne, n), u, titer)

                # compute forward tomography operator for further updates of rho and lambda
                h = unpaddata(tslv.fwd_tomo_batch(u), ne, n)
                # 3. dual update
                lamd = lamd+rho*(h-psi)

                if(np.mod(k, 4) == 0):  # check Lagrangian, save current iteration results
                    Dpsi = dslv.apply_flow_gpu_batch(psi, flow)
                    lagr[k, 0] = 0.5*np.linalg.norm(Dpsi-data)**2
                    lagr[k, 1] = np.sum(lamd*(h-psi))
                    lagr[k, 2] = 0.5*rho*np.linalg.norm(h-psi)**2
                    lagr[k, 2] = 0.5*rho*np.linalg.norm(h-psi)**2
                    lagr[k, 3] = np.sum(lagr[k, 0:3])
                    print("iter %d, %.2f wsize %d, rho %.2f, Lagrangian %.4e %.4e %.4e Total %.4e " % (
                        k, np.linalg.norm(flow), pars[2], rho, *lagr[k]))
                    sys.stdout.flush()
                    # save object
                    dxchange.write_tiff_stack(unpadobject(
                        u, ne, n),  fname+'/data/of_recon/recon/iter'+str(k), overwrite=True)
                    # save flow figure
                    np.save(fname+'/data/of_recon/flow'+str(k), flow)
                dslv.flowplot(
                    u, psi, flow, fname+'/data/of_recon/flow_iter'+str(k))
                # update rho
                rho = _update_penalty(psi, h, h0, rho)
                h0 = h

                if(pars[2] > 12):  # limit optical flow window size
                    pars[2] -= stepwin
        res['u'] = u
        res['psi'] = psi
        res['h0'] = h0
        res['lamd'] = lamd
        res['flow'] = flow
        res['lagr'] = lagr

    return res


def admm_of_reg(data, theta, pnz, ptheta, center, alpha, ngpus, niter, startwin, stepwin, res=None, fname=''):
    """Reconstruct with the optical flow method and regularization (OFTV)"""

    [ntheta, nz, n] = data.shape
    ne = _take_psize(n)
    # tomographic solver on GPU
    lagr = np.zeros([niter, 6], dtype='float32')
    with SolverTomo(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus) as tslv:
        # alignment solver on GPU
        with SolverDeform(ntheta, nz, n, ptheta, ngpus) as dslv:
            # find min,max values accoding to histogram
            mmin, mmax = find_min_max(data)
            # initial guess and coordinating v ariables
            if(res == None):
                u = np.zeros([nz, ne, ne], dtype='float32')
                psi1 = data.copy()
                psi2 = np.zeros([2, nz, ne, ne], dtype='float32')
                h01 = psi1.copy()
                h02 = psi2.copy()
                lamd1 = np.zeros([ntheta, nz, n], dtype='float32')
                lamd2 = np.zeros([2, nz, ne, ne], dtype='float32')
                flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
                res = {}
            else:
                u = res['u']
                psi1 = res['psi1']
                psi2 = res['psi2']
                h01 = res['h01']
                h02 = res['h02']
                lamd1 = res['lamd1']
                lamd2 = res['lamd2']
                flow = res['flow']
            # optical flow parameters (see openCV function for Farneback's algorithm)
            pars = [0.5, 1, startwin, 4, 5, 1.1, 4]
            rho1 = 0.5  # weighting factor in ADMM w.r.t. tomo sub-problem
            rho2 = 0.5  # weighting factor in ADMM w.r.t. deform sub-problem

            for k in range(niter):

                # 1. Solve the alignment sub-problem
                # register flow
                flow = dslv.registration_flow_batch(
                    psi1, data, mmin, mmax, flow, pars)
                # unwarping
                psi1 = dslv.cg_deform_gpu_batch(data, psi1, flow, 4, unpaddata(
                    tslv.fwd_tomo_batch(u), ne, n)+lamd1/rho1, rho1)

                # 2. Solve the regularization sub-problen
                psi2 = tslv.solve_reg(u, lamd2, rho2, alpha)

                # 3. Solve the tomography sub-problen
                u = tslv.cg_tomo_reg_batch(
                    paddata(psi1-lamd1/rho1, ne, n), u, 4, rho2/rho1, psi2-lamd2/rho2)

                # compute forward operators for further updates of rho and lamd
                h1 = unpaddata(tslv.fwd_tomo_batch(u), ne, n)
                h2 = tslv.fwd_reg(u)

                # 4. dual update
                lamd1 = lamd1+rho1*(h1-psi1)
                lamd2 = lamd2+rho2*(h2-psi2)

                if(np.mod(k, 4) == 0):  # check Lagrangian, save current iteration results
                    Dpsi1 = dslv.apply_flow_gpu_batch(psi1, flow)
                    lagr[k, 0] = 0.5*np.linalg.norm(Dpsi1-data)**2
                    lagr[k, 1] = np.sum(lamd1*(h1-psi1))
                    lagr[k, 2] = 0.5*rho1*np.linalg.norm(h1-psi1)**2
                    lagr[k, 3] = 0.5*rho2*np.linalg.norm(h2-psi2)**2
                    lagr[k, 4] = np.sum(np.sqrt(np.sum(psi2**2, 0)))
                    lagr[k, 5] = np.sum(lagr[k, 0:5])
                    print("iter %d, flow %.2f wsize %d, rho (%.2f,%.2f), Lagrangian terms %.4e %.4e %.4e %.4e %.4e Total %.4e " % (
                        k, np.linalg.norm(flow), pars[2], rho1, rho2, *lagr[k]))
                    sys.stdout.flush()
                    # save object
                    dxchange.write_tiff(unpadobject(
                        u, ne, n),  fname+'/data/of_recon_reg/recon/iter'+str(k), overwrite=True)
                    # save flow figure
                    dslv.flowplot(
                        u, psi1, flow, fname+'/data/of_recon_reg/flowiter'+str(k))

                # update rho
                rho1 = _update_penalty(psi1, h1, h01, rho1)
                rho2 = _update_penalty(psi2, h2, h02, rho2)

                h01 = h1
                h02 = h2

                if(pars[2] > 12):  # limit optical flow window size
                    pars[2] -= stepwin

    res['u'] = u
    res['psi1'] = psi1
    res['psi2'] = psi2
    res['h01'] = h01
    res['h02'] = h02
    res['lamd1'] = lamd1
    res['lamd2'] = lamd2
    res['flow'] = flow

    return res


def _update_penalty(psi, h, h0, rho):
    """Update the ADMM weighting penalty for faster convergence"""

    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho*(h-h0))**2
    if (r > 10*s):
        rho *= 2
    elif (s > 10*r):
        rho *= 0.5
    return rho


def _downsample(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[:, ::2]+res[:, 1::2])
        res = 0.5*(res[:, :, ::2]+res[:, :, 1::2])
    return res


def _fftupsample(f, dims):
    paddim = np.zeros([np.ndim(f), 2], dtype='int32')
    dims = np.asarray(dims).astype('int32')
    paddim[dims, 0] = np.asarray(f.shape)[dims]//2
    paddim[dims, 1] = np.asarray(f.shape)[dims]//2
    fsize = f.size
    f = sp.fft.ifftshift(sp.fft.fftn(sp.fft.fftshift(
        f, dims), axes=dims, workers=-1, overwrite_x=True), dims)
    f = np.pad(f, paddim)
    f = sp.fft.ifftshift(sp.fft.ifftn(sp.fft.fftshift(
        f, dims), axes=dims, workers=-1, overwrite_x=True), dims)
    return f.real.astype('float32')*(f.size/fsize)


def _upsample(init):
    # init['u'] = ndimage.zoom(init['u'], 2, order=1)
    # init['psi'] = ndimage.zoom(init['psi'], (1, 2, 2), order=1)
    # init['h0'] = ndimage.zoom(init['h0'], (1, 2, 2), order=1)
    # init['lamd'] = ndimage.zoom(init['lamd'], (1, 2, 2), order=1)
    # init['flow'] = ndimage.zoom(init['flow'], (1, 2, 2, 1), order=1)*2

    init['u'] = _fftupsample(init['u'], [0, 1, 2])
    init['psi'] = _fftupsample(init['psi'], [1, 2])
    init['h0'] = _fftupsample(init['h0'], [1, 2])
    init['lamd'] = _fftupsample(init['lamd'], [1, 2])
    init['flow'] = _fftupsample(init['flow'], [1, 2])*2

    return init


def _upsample_reg(init):

    init['u'] = _fftupsample(init['u'], [0, 1, 2])
    init['psi1'] = _fftupsample(init['psi1'], [1, 2])
    init['psi2'] = _fftupsample(init['psi2'], [1, 2, 3])

    init['h01'] = _fftupsample(init['h01'], [1, 2])
    init['h02'] = _fftupsample(init['h02'], [1, 2, 3])

    init['lamd1'] = _fftupsample(init['lamd1'], [1, 2])
    init['lamd2'] = _fftupsample(init['lamd2'], [1, 2, 3])

    init['flow'] = _fftupsample(init['flow'], [1, 2])*2

    return init


def _take_psize(n):
    s = bin(int(3*n//2))
    s = s[:5]+s[5:].replace('1', '0')
    ne = int(s, 2)
    # s = bin(int(3*n//2))
    # s = s[:5]+s[5:].replace('1', '0')
    # ne = int(s, 2)
    
    # ne=n
    print('padded size', ne)
    return ne


def admm_of_levels(data, theta, pnz, ptheta, center, ngpus, niter, startwin, stepwin, fname):
    res = None
    levels = len(niter)
    for k in np.arange(0, levels):
        databin = _downsample(data, levels-k-1)
        # res = admm_of(databin, theta, int(pnz/pow(2, levels-k-1)), ptheta, center/pow(
            # 2, levels-k-1), ngpus, niter[k], startwin[k], stepwin[k], res, fname)
        res = admm_of(databin, theta, int(np.ceil(pnz/pow(2, levels-k-1))), ptheta, center/pow(
            2, levels-k-1), ngpus, niter[k], startwin[k], stepwin[k], res, fname)
        
        if(k < levels-1):
            res = _upsample(res)

    return res


def admm_of_reg_levels(data, theta, pnz, ptheta, center, alpha, ngpus, niter, startwin, stepwin, fname):
    res = None
    levels = len(niter)
    for k in np.arange(0, levels):
        databin = _downsample(data, levels-k-1)
        res = admm_of_reg(databin, theta, int(pnz/pow(2, levels-k-1)), ptheta, center/pow(
            2, levels-k-1), alpha/pow(2, levels-k-1), ngpus, niter[k], startwin[k], stepwin[k], res, fname)
        if(k < levels-1):
            res = _upsample_reg(res)

    return res
