import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import elasticdeform
import concurrent.futures as cf
import threading
import matplotlib.pyplot as plt
import os
from itertools import repeat
from functools import partial


def gencylinder(nz, n):
    [x, y, z] = np.meshgrid(np.arange(-n/2, n/2)/n*2,
                            np.arange(-nz/2, nz/2)/nz*2, np.arange(-n/2, n/2)/n*2)
    a = 0.5
    b = 0.7
    alpha = 0
    beta = 1.4
    x1 = x*np.cos(alpha)+y*np.sin(alpha)
    y1 = x*np.sin(alpha)-y*np.cos(alpha)
    y2 = y1*np.cos(beta)+z*np.sin(beta)
    z2 = y1*np.sin(beta)-z*np.cos(beta)

    u1 = (x1)**2/a**2+(y2)**2/b**2 < 0.45
    u2 = (x1)**2/(a*0.95)**2+(y2)**2/(b*0.95)**2 < 0.4
    u3 = (x1)**2/(a*0.9)**2+(y2)**2/(b*0.9)**2 < 0.4
    u4 = (x1)**2/(a*0.85)**2+(y2)**2/(b*0.85)**2 < 0.4
    u1 = sp.ndimage.gaussian_filter(np.float32(u1), 0.5)
    u2 = sp.ndimage.gaussian_filter(np.float32(u2), 0.5)
    u3 = sp.ndimage.gaussian_filter(np.float32(u3), 0.5)
    u4 = sp.ndimage.gaussian_filter(np.float32(u4), 0.5)
    u = u1-u2+u3-u4
    u[0:32, :] = 0
    u[-32:, :] = 0
    return u


def deform_data(data_deform, u, theta, n, nz, center, displacement, start, i):
    with tc.SolverTomo(theta[i:i+1], 1, nz, n, 1, center) as slv:
        # generate data
        u_deform = elasticdeform.deform_grid(u, displacement*(
            i-start+1)/ntheta*4, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
        data_deform[i] = slv.fwd_tomo_batch(
            u_deform.astype('complex64'))
    return data_deform[i]


def deform_data_batch(u, theta, ntheta, n, nz, center):
    points = [3, 3, 3]
    sigma = 20
    # np.random.seed(0)
    displacement = (np.random.rand(3, *points) - 0.5) * sigma
    start = ntheta//2
    end = ntheta//2+ntheta//8
    data_deform = np.zeros((ntheta, nz, n), dtype='complex64')

    u_deform = elasticdeform.deform_grid(u, displacement*(
        end-start)/ntheta*4, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
    for i in range(0, start):
        with tc.SolverTomo(theta[i:i+1], 1, nz, n, 1, center) as slv:
            data_deform[i] = slv.fwd_tomo_batch(u.astype('complex64'))
    for i in range(end, ntheta):
        with tc.SolverTomo(theta[i:i+1], 1, nz, n, 1, center) as slv:
            data_deform[i] = slv.fwd_tomo_batch(
                u_deform.astype('complex64'))
    with cf.ThreadPoolExecutor() as e:
        shift = start
        for res0 in e.map(partial(deform_data, data_deform, u, theta, n, nz, center, displacement, start), range(start, end)):
            data_deform[shift] = res0
            shift += 1

    return data_deform


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
    plt.clf()

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

    # Model parameters
    n = 256  # object size n x,y
    nz = 128  # object size in z
    ntheta = 128*3  # number of angles (rotations)
    center = n/2  # rotation center
    theta = np.linspace(0, 3*np.pi, ntheta).astype('float32')  # angles

    niter = 64  # tomography iterations
    pnz = 128  # number of slice partitions for simultaneous processing in tomography

    u0 = gencylinder(nz, n)
    # Deform data and save to file
    data_deform = deform_data_batch(u0, theta, ntheta, n, nz, center)
    dxchange.write_tiff(data_deform.real, 'dataret', overwrite=True)
    dxchange.write_tiff(data_deform.imag, 'dataimt', overwrite=True)
    # or load from file
    # data_deform = dxchange.read_tiff('dataret.tiff').astype(
        # 'float32')+1j*dxchange.read_tiff('dataimt.tiff').astype('float32')

    # data
    data = data_deform.copy()

    # initial guess
    u = np.zeros([nz, n, n], dtype='complex64')
    psi = data.copy()
    lamd = np.zeros([ntheta, nz, n], dtype='complex64')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    # optical flow parameters
    pars = [0.5, 3, 128, 16, 5, 1.1, 0]

    # ADMM solver
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center) as tslv:
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
                        u.real,  'tmp'+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    dxchange.write_tiff_stack(
                        psi.real, 'tmp'+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                pars[2] -= 1
