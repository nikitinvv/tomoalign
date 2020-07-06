"""Module for tomography."""

import cupy as cp
import numpy as np
from .radonusfft import radonusfft
import threading
import concurrent.futures as cf
from functools import partial


class SolverTomo(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of slice partitions to process together
        simultaneously.
    ngpus : int
        Number of gpus        
    """

    def __init__(self, theta, ntheta, nz, n, pnz, center, ngpus):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        # print(theta, ntheta, nz, ne, pnz, center+(ne-n)/2, ngpus)
    
        super().__init__(ntheta, pnz, n, center, theta.ctypes.data, ngpus)
        self.nz = nz        
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u, gpu):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        u0 = u.astype('complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u0.data.ptr, gpu)        
        return res.real

    def adj_tomo(self, data, gpu):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        data0 = data.astype('complex64')
        # C++ wrapper, send pointers to GPU arrays        
        self.adj(res.data.ptr, data0.data.ptr, gpu)
        return res.real

    def line_search(self, minf, gamma, Ru, Rd):
        """Line search for the step sizes gamma"""
        while(minf(Ru)-minf(Ru+gamma*Rd) < 0):
            gamma *= 0.5
        return gamma

    def line_search_ext(self, minf, gamma, Ru, Rd, gu, gd):
        """Line search for the step sizes gamma"""
        while(minf(Ru, gu)-minf(Ru+gamma*Rd, gu+gamma*gd) < 0):
            gamma *= 0.5
        return gamma        
    
    def fwd_tomo_batch(self, u):
        """Batch of Tomography transform (R)"""
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='float32')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu,0)
            # copy result to cpu
            res[:, ids] = res_gpu.get()
        return res

    def adj_tomo_batch(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = np.zeros([self.nz, self.n, self.n], dtype='complex64')
        for k in range(0, self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            data_gpu = cp.array(data[:, ids])

            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu,0)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def fwd_reg(self, u):
        """Forward operator for regularization (J)"""
        res = cp.get_array_module(u).zeros([2, *u.shape], dtype='float32')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        # res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        return res

    def adj_reg(self, gr):
        """Adjoint operator for regularization (J*)"""
        res = cp.get_array_module(gr).zeros(gr.shape[1:], dtype='float32')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        # res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        # res[0, :, :] += gr[2, 0, :, :]
        return -res
        
    def solve_reg(self, u, mu, tau, alpha):
        """Solution of the L1 problem by soft-thresholding"""
        z = self.fwd_reg(u)+mu/tau
        za = np.sqrt(np.real(np.sum(z*np.conj(z), 0)))
        z[:, za <= alpha/tau] = 0
        z[:, za > alpha/tau] -= alpha/tau * \
            z[:, za > alpha/tau]/(za[za > alpha/tau])
        return z        

    def cg_tomo(self, xi0, u, titer,gpu=0, dbg=False):
        """CG solver for 1 slice partition"""
        # minimization functional
        def minf(Ru):
            f = cp.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo(u,gpu)            
            grad = self.adj_tomo(Ru-xi0,gpu) / \
                (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo(d,gpu)
            gamma = 0.5#*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru)))
        return u
    
     

    def cg_tomo_multi_gpu(self,xi0,u,titer,lock,ids):
        """Pick GPU, copy data, run reconstruction"""
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        u_gpu = cp.array(u[ids])
        xi0_gpu = cp.array(xi0[:, ids])
        # reconstruct
        u_gpu = self.cg_tomo(xi0_gpu, u_gpu, titer, gpu)
        u[ids] = u_gpu.get()

        BUSYGPUS[gpu] = 0

        return u[ids]
  
    def cg_tomo_batch(self, xi0, init, titer, dbg=False):
        """CG solver for rho||Ru-xi0||_2 by z-slice partitions"""
        u = init.copy()
        ids_list = [None]*int(np.ceil(self.nz/float(self.pnz)))
        for k in range(0, len(ids_list)):
            ids_list[k] = range(k*self.pnz, min(self.nz, (k+1)*self.pnz))
        
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.cg_tomo_multi_gpu, xi0, u, titer, lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]
        cp.cuda.Device(0).use()        
        return u


    def cg_tomo_reg(self, xi0, u, titer, tau, xi1,gpu=0, dbg=False):
        """CG solver for 1 slice partition"""
        # minimization functional
        def minf(Ru,gu):
            f = cp.linalg.norm(Ru-xi0)**2+tau*cp.linalg.norm(gu-xi1)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo(u,gpu)            
            gu = self.fwd_reg(u)   
            grad = (self.adj_tomo(Ru-xi0,gpu) / \
                (self.ntheta * self.n/2)+tau*self.adj_reg(gu-xi1))/2/max(tau,1)# normalized gradient
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Rd = self.fwd_tomo(d,gpu)
            gd = self.fwd_reg(d)
            gamma = 0.5#*self.line_search_ext(minf, 1, Ru, Rd,gu,gd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(Ru,gu)))
        return u
    
     

    def cg_tomo_reg_multi_gpu(self,xi0,u,titer,tau,xi1,lock,ids):
        """Pick GPU, copy data, run reconstruction"""    
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        u_gpu = cp.array(u[ids])
        xi0_gpu = cp.array(xi0[:, ids])
        xi1_gpu = cp.array(xi1[:, ids])
        # reconstruct
        u_gpu = self.cg_tomo_reg(xi0_gpu, u_gpu, titer, tau, xi1_gpu, gpu)
        u[ids] = u_gpu.get()

        BUSYGPUS[gpu] = 0

        return u[ids]
  
    # Conjugate gradients tomography (by slices partitions)
    def cg_tomo_reg_batch(self, xi0, init, titer, tau, xi1, dbg=False):
        """CG solver for ||Ru-xi0||_2+tau||Ju-xi1||_2 by z-slice partitions"""
        u = init.copy()
        ids_list = [None]*int(np.ceil(self.nz/float(self.pnz)))
        for k in range(0, len(ids_list)):
            ids_list[k] = range(k*self.pnz, min(self.nz, (k+1)*self.pnz))
        
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.cg_tomo_reg_multi_gpu, xi0, u, titer, tau, xi1, lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]
        cp.cuda.Device(0).use()        
        return u

