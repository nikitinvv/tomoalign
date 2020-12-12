import numpy as np
import concurrent.futures as cf
import threading
from functools import partial
import cv2
import cupy as cp
import matplotlib.pyplot as plt
from .deform import deform
from .flowvis import flow_to_color
import matplotlib
matplotlib.use('Agg')

class SolverDeform(deform):
    """Base class for deformation solvers.
    This class is a context manager which provides the basic operators required
    to implement an alignment solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.   
    """

    def __init__(self, ntheta, nz, n, ptheta, ngpus):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, nz, n, ptheta, ngpus)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def apply_flow_cpu(self, f, flow, id):
        """Apply optical flow for one projection."""
        flow0 = flow[id].copy()
        h, w = flow0.shape[:2]
        flow0 = -flow0
        flow0[:, :, 0] += np.arange(w)
        flow0[:, :, 1] += np.arange(h)[:, np.newaxis]
        f0 = f[id]
        res = cv2.remap(f0, flow0,
                        None, cv2.INTER_LANCZOS4)
        return res

    def apply_flow_cpu_batch(self, psi, flow, nproc=16):
        """Apply optical flow for all projection in parallel."""
        res = np.zeros(psi.shape, dtype='float32')
        with cf.ThreadPoolExecutor(nproc) as e:
            shift = 0
            for res0 in e.map(partial(self.apply_flow_cpu, psi, flow), range(0, psi.shape[0])):
                res[shift] = res0
                shift += 1
        return res

    def registration_flow(self, psi, g, mmin, mmax, flow, pars, id):
        """Find optical flow for one projection"""
        tmp1 = ((psi[id]-mmin[id]) /
                (mmax[id]-mmin[id])*255)
        tmp1[tmp1 > 255] = 255
        tmp1[tmp1 < 0] = 0
        tmp2 = ((g[id]-mmin[id]) /
                (mmax[id]-mmin[id])*255)
        tmp2[tmp2 > 255] = 255
        tmp2[tmp2 < 0] = 0
        tmp1=np.uint8(tmp1)
        tmp2=np.uint8(tmp2)
        
        flow[id]=cv2.calcOpticalFlowFarneback(
           tmp1, tmp2, flow[id], *pars)  # updates flow
        #if(cp.linalg.norm(flow[id])!=0):
         #    print(np.linalg.norm(flow[id]),cp.linalg.norm(tmp1-tmp2))
         

    def registration_flow_batch(self, psi, g, mmin, mmax, flow=None, pars=[0.5, 3, 20, 16, 5, 1.1, 4]):
        """Find optical flow for all projections in parallel"""
        if (flow is None):
            flow = np.zeros([self.ntheta, self.nz, self.n, 2], dtype='float32')
        total = 0
        for k in range(self.ntheta//self.ptheta//10):
            ids = np.arange(k*self.ptheta*10,(k+1)*self.ptheta*10)
            flownew = flow[ids].copy()
            with cf.ThreadPoolExecutor(20) as e:
                # update flow in place
                e.map(partial(self.registration_flow, psi[ids], g[ids], mmin[ids],
                          mmax[ids], flownew, pars), range(0, len(ids)))

            # control Farneback's (may diverge for small window sizes)
            err = np.linalg.norm(g[ids]-self.apply_flow_gpu_batch(psi[ids], flownew),axis=(1,2))
            err1 = np.linalg.norm(g[ids]-self.apply_flow_gpu_batch(psi[ids], flow[ids]),axis=(1,2))
            idsgood = np.where(err1>=err)[0]        
            total+=len(idsgood)
            flow[ids[idsgood]] = flownew[idsgood]
        print('bad alignment for:',self.ntheta-total)
        return flow

    def line_search(self, minf, gamma, psi, Dpsi, d, Td):
        """Line search for the step sizes gamma"""
        while(minf(psi, Dpsi)-minf(psi+gamma*d, Dpsi+gamma*Td) < 0):
            gamma *= 0.5
        if(gamma<0.125):
            gamma = 0
        return gamma

    
    def apply_flow_gpu(self, f, flow, gpu):
        h, w = flow.shape[1:3]
        flow = -flow.copy()
        flow[:, :, :, 0] += cp.arange(w)
        flow[:, :, :, 1] += cp.arange(h)[:, cp.newaxis]

        flowx = cp.asarray(flow[:, :, :, 0], order='C')
        flowy = cp.asarray(flow[:, :, :, 1], order='C')
        g = f.copy()  # keep values that were not affected
        # g = cp.zeros([self.ptheta,self.nz,self.n],dtype='float32')

        self.remap(g.data.ptr, f.data.ptr, flowx.data.ptr, flowy.data.ptr, gpu)
        return g

    def apply_flow_gpu_batch(self, f, flow):
        res = np.zeros([f.shape[0], self.nz, self.n], dtype='float32')
        for k in range(0, f.shape[0]//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            f_gpu = cp.array(f[ids])
            flow_gpu = cp.array(flow[ids])
            # Radon transform
            res_gpu = self.apply_flow_gpu(f_gpu, flow_gpu, 0)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def cg_deform(self, data, psi, flow, titer, xi1=0, rho=0, gpu=0):
        """CG solver for deformation"""
        # minimization functional
        def minf(psi, Dpsi):
            f = cp.linalg.norm(Dpsi-data)**2+rho*cp.linalg.norm(psi-xi1)**2
            return f

        for i in range(titer):
            Dpsi = self.apply_flow_gpu(psi, flow, gpu)
            grad = (self.apply_flow_gpu(Dpsi-data, -flow, gpu) +
                    rho*(psi-xi1))/max(rho, 1)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Td = self.apply_flow_gpu(d, flow, gpu)
            gamma = 0.5*self.line_search(minf, 1, psi, Dpsi, d, Td)
            if(gamma==0):
                break
            grad0 = grad
            # update step
            psi = psi + gamma*d
            # check convergence
            # if (0):
            #     print("%4d, %.3e, %.7e" %
            #           (i, gamma, minf(psi, Dpsi+gamma*Td)))
        return psi

    def cg_deform_multi_gpu(self, data, psi, flow,  titer, xi1, rho, lock, ids):
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

        data_gpu = cp.array(data[ids])
        psi_gpu = cp.array(psi[ids])
        xi1_gpu = cp.array(xi1[ids])
        flow_gpu = cp.array(flow[ids])
        # Radon transform
        psi_gpu = self.cg_deform(
            data_gpu, psi_gpu, flow_gpu, titer, xi1_gpu, rho, gpu)
        # copy result to cpu
        psi[ids] = psi_gpu.get()

        BUSYGPUS[gpu] = 0

        return psi[ids]

    def cg_deform_gpu_batch(self, data, init, flow, titer, xi1=0, rho=0, dbg=False):

        psi = init.copy()
        ids_list = [None]*int(np.ceil(self.ntheta/float(self.ptheta)))
        for k in range(0, len(ids_list)):
            ids_list[k] = range(
                k*self.ptheta, min(self.ntheta, (k+1)*self.ptheta))

        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for psii in e.map(partial(self.cg_deform_multi_gpu, data, psi, flow, titer, xi1, rho, lock), ids_list):
                psi[np.arange(0, psii.shape[0])+shift] = psii
                shift += psii.shape[0]
        cp.cuda.Device(0).use()

        return psi

    def flowplot(self, u, psi, flow, fname):
        """Visualize 4 aligned projections, corrsponding optical flows, 
        and slices through reconsruction, save figure as a png file"""

        [ntheta, nz, n] = psi.shape

        plt.figure(figsize=(10, 7))
        plt.subplot(3, 4, 1)
        plt.imshow(psi[ntheta//4], cmap='gray')

        plt.subplot(3, 4, 2)
        plt.imshow(psi[ntheta//2], cmap='gray')
        plt.subplot(3, 4, 3)
        plt.imshow(psi[3*ntheta//4], cmap='gray')

        plt.subplot(3, 4, 4)
        plt.imshow(psi[-1].real, cmap='gray')

        plt.subplot(3, 4, 5)
        plt.imshow(flow_to_color(flow[ntheta//4]), cmap='gray')

        plt.subplot(3, 4, 6)
        plt.imshow(flow_to_color(flow[ntheta//2]), cmap='gray')

        plt.subplot(3, 4, 7)
        plt.imshow(flow_to_color(flow[3*ntheta//4]), cmap='gray')
        plt.subplot(3, 4, 8)
        plt.imshow(flow_to_color(flow[-1]), cmap='gray')

        plt.subplot(3, 4, 9)
        plt.imshow(u[nz//2], cmap='gray')
        plt.subplot(3, 4, 10)
        plt.imshow(u[nz//2+nz//8], cmap='gray')

        plt.subplot(3, 4, 11)
        plt.imshow(u[:, n//2-35], cmap='gray')

        plt.subplot(3, 4, 12)
        plt.imshow(u[:, :, n//2-35], cmap='gray')
        plt.savefig(fname)
        plt.close()
