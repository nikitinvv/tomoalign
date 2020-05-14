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
import gc
import scipy.ndimage as ndimage
matplotlib.use('Agg')
centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run1_8keV_phase_interlaced_100prj_per_rot_1201prj_1s_006': 1204,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run21_40min_8keV_phase_interlaced_1201prj_1s_012': 1187,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_PBI_new_ROI_8keV_phase_interlaced_2000prj_1s_042':  1250,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_ROI2_8keV_phase_interlaced_1201prj_0.5s_037':  1227,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_PBI_new_ROI_8keV_phase_interlaced_1201prj_0.5s_041': 1248,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024': 1202}
ngpus = 4
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
    plt.imshow(u[nz//2], cmap='gray')
    plt.subplot(3, 4, 10)
    plt.imshow(u[nz//2+nz//8], cmap='gray')

    plt.subplot(3, 4, 11)
    plt.imshow(u[:, n//2], cmap='gray')

    plt.subplot(3, 4, 12)
    plt.imshow(u[:, :, n//2], cmap='gray')
    if not os.path.exists(name+'/flowfw_'+'_'+str(ntheta)):
        os.makedirs(
            name+'/flowfw_'+'_'+str(ntheta))
    plt.savefig(
        name+'/flowfw_'+'_'+str(ntheta)+'/'+str(k))
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
        
def find_min_max(data):
    # s = np.std(data,axis=(1,2))    
    # m = np.mean(data,axis=(1,2))
    # mmin = m-2*s
    # mmax = m+2*s
    mmin = np.zeros(data.shape[0],dtype='float32')
    mmax = np.zeros(data.shape[0],dtype='float32')
    
    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:],1000)
        stend = np.where(h>np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]        
        mmin[k] = e[st]
        mmax[k] = e[end+1]
     
    return mmin,mmax

def pad(data,ne,n):
    datae = np.zeros([data.shape[0],nz,ne],dtype='float32')
    datae[:,:,ne//2-n//2:ne//2+n//2]=data
    datae[:,:,:ne//2-n//2]=datae[:,:,ne//2-n//2:ne//2-n//2+1]
    datae[:,:,ne//2+n//2:]=datae[:,:,ne//2+n//2-1:ne//2+n//2]
    return datae

def unpad(data,ne,n):
    return data[:,:,ne//2-n//2:ne//2+n//2]

def interpdense(u,psi,lamd,flow):
    [ntheta,nz,n]=psi.shape
    #u
    # fu=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(u)))
    # fue = np.zeros([2*nz,2*n,2*n],dtype='complex64')
    # fue[nz//2:-nz//2,n//2:-n//2,n//2:-n//2]=fu
    # u = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(fue))).real*8
    # #
    # [ntheta,nz,n]=psi.shape
    
    # fpsi=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psi)))
    # fpsie = np.zeros([ntheta,2*nz,2*n],dtype='complex64')
    # fpsie[:,nz//2:-nz//2,n//2:-n//2]=fpsi
    # psi = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fpsie))).real*4
    # #
    # flamd=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(lamd)))
    # flamde = np.zeros([ntheta,2*nz,2*n],dtype='complex64')
    # flamde[:,nz//2:-nz//2,n//2:-n//2]=flamd
    # lamd = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(flamde))).real*4
    # #
    # flownew = np.zeros([ntheta,2*nz,2*n,2],dtype='float32')    
    # fflow=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(flow[:,:,:,0])))
    # fflowe = np.zeros([ntheta,2*nz,2*n],dtype='complex64')
    # fflowe[:,nz//2:-nz//2,n//2:-n//2]=fflow
    # flownew[:,:,:,0] = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fflowe))).real*4

    # fflow=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(flow[:,:,:,1])))
    # fflowe = np.zeros([ntheta,2*nz,2*n],dtype='complex64')
    # fflowe[:,nz//2:-nz//2,n//2:-n//2]=fflow
    # flownew[:,:,:,1] = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fflowe))).real*4

    unew = ndimage.zoom(u,2)
    psinew = np.zeros([ntheta,2*nz,2*n],dtype='float32')    
    lamdnew = np.zeros([ntheta,2*nz,2*n],dtype='float32')        
    flownew = np.zeros([ntheta,2*nz,2*n,2],dtype='float32')    
    for k in range(ntheta):
        psinew[k] = ndimage.zoom(psi[k],2)
        lamdnew[k] = ndimage.zoom(lamd[k],2)
        flownew[k,:,:,0]=ndimage.zoom(flow[k,:,:,0],2,order=0)
        flownew[k,:,:,1]=ndimage.zoom(flow[k,:,:,1],2,order=0)

    return unew,psinew,lamdnew,flownew

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]   
    
    w = [256,128,64]
    niter = [96,48,24]
    binnings=[3,2,1]
    # ADMM solver
    for il in range(3):
        binning = binnings[il]
        data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
        theta = np.zeros(ndsets*nth,dtype='float32')
        for k in range(ndsets):
            data[k*nth:(k+1)*nth] = np.load(name+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
            theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
            [ntheta, nz, n] = data.shape  # object size n x,y
        [ntheta, nz, n] = data.shape  # object size n x,y
   
        data[np.isnan(data)]=0            
        data-=np.mean(data)
        mmin,mmax = find_min_max(data)
        # pad data    
        ne = 3456//pow(2,binning)    
        #ne=n
        center = centers[sys.argv[3]]+(ne//2-n//2)*pow(2,binning)        
        pnz = 8*pow(2,binning)  # number of slice partitions for simultaneous processing in tomography
        ptheta = 50
        # initial guess

        if(binning==3):
            u = np.zeros([nz, ne, ne], dtype='float32')
            psi = data.copy()
            lamd = np.zeros([ntheta, nz, n], dtype='float32')
            flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
        else:            
            #print(flow[0,:4,:4,:])           
            u, psi, lamd, flow = interpdense(u,psi,lamd,flow)
           # print(flow[0,:4,:4,:])  
            # print(np.linalg.norm(u))
            # print(np.linalg.norm(u1[::2,::2,::2]-u))
            # print(np.linalg.norm(psi1[:,::2,::2]-psi))
            # print(np.linalg.norm(lamd1[:,::2,::2]-lamd))
            # print(np.linalg.norm(flow1[:,::2,::2]-flow))
        
        # optical flow parameters
        pars = [0.5,0, w[il], 4, 5, 1.1, 4]
        with tc.SolverTomo(theta, ntheta, nz, ne, pnz, center/pow(2, binning), ngpus) as tslv:
             with dc.SolverDeform(ntheta, nz, n, ptheta) as dslv:
                rho = 0.5
                h0 = psi
                for k in range(niter[il]):
                    # registration
                    flow = dslv.registration_flow_batch(
                        psi, data, mmin, mmax, flow.copy(), pars, 20) 
                    
                    # deformation subproblem
                    psi = dslv.cg_deform_gpu_batch(data, psi, flow, 4,
                                                unpad(tslv.fwd_tomo_batch(u),ne,n)+lamd/rho, rho)
                    # tomo subproblem
                    u = tslv.cg_tomo_batch(pad(psi-lamd/rho,ne,n), u, 4)
                    h = unpad(tslv.fwd_tomo_batch(u),ne,n)
                    # lambda update
                    lamd = lamd+rho*(h-psi)

                    # checking intermediate results
                    myplot(u, psi, flow, binning)
                    if(np.mod(k,4)==0):  # check Lagrangian
                        Tpsi = dslv.apply_flow_gpu_batch(psi, flow)
                        lagr = np.zeros(4)
                        lagr[0] = 0.5*np.linalg.norm(Tpsi-data)**2
                        lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                        lagr[2] = rho/2*np.linalg.norm(h-psi)**2
                        lagr[3] = np.sum(lagr[0:3])
                        print("%d %d %.2e %.2f %.4e %.4e %.4e %.4e " % (k, pars[2], np.linalg.norm(
                            flow), rho, *lagr))
                        dxchange.write_tiff_stack(
                            u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  name+'/fw_'+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                        dxchange.write_tiff_stack(
                        psi.real, name+'/fw_'+'_'+str(ntheta)+'/psi'+str(k)+'/r', overwrite=True)

                    # Updates
                    rho = update_penalty(psi, h, h0, rho)
                    h0 = h
                    pars[2] -= 2
                    sys.stdout.flush()
                    gc.collect()