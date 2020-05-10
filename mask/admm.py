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

#Run1_8keV_phase_interlaced_100prj_per_rot_1201prj_1s_006.h5 center 1204
#Run21_40min_8keV_phase_interlaced_1201prj_1s_012 center 1187
#PAN_PI_PBI_new_ROI_8keV_phase_interlaced_2000prj_1s_042 center 1250
#PAN_PI_ROI2_8keV_phase_interlaced_1201prj_0.5s_037.h5 center 1227
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
    if not os.path.exists(name+'/flowfw_'+str(binning)+'_'+str(ntheta)):
        os.makedirs(
            name+'/flowfw_'+str(binning)+'_'+str(ntheta))
    plt.savefig(
        name+'/flowfw_'+str(binning)+'_'+str(ntheta)+'/'+str(k))
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
        #print(mmin[st],mmax[end+1])        
    return mmin,mmax

if __name__ == "__main__":
    binning = 1
    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]
    data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(name+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
    [ntheta, nz, n] = data.shape  # object size n x,y
   
    data[np.isnan(data)]=0    
    data-=np.mean(data)
    mmin,mmax = find_min_max(data)
    print(mmin,mmax)
    # pad data    
    ne = 3456//pow(2,binning)
    #ne=n
    print(ne)
    datae = np.zeros([ntheta,nz,ne],dtype='float32')
    datae[:,:,ne//2-n//2:ne//2+n//2]=data
    datae[:,:,:ne//2-n//2]=datae[:,:,ne//2-n//2:ne//2-n//2+1]
    datae[:,:,ne//2+n//2:]=datae[:,:,ne//2+n//2-1:ne//2+n//2]
    data=datae
    center = 1187+(ne//2-n//2)*pow(2,binning)
    n=ne
    print('shape',data.shape,'center',center//pow(2,binning))
    
    #dxchange.write_tiff_stack(data,name+'/data')
    niter = 128  # tomography iterations
    pnz = 4*pow(2,binning)  # number of slice partitions for simultaneous processing in tomography
    ptheta = 25*pow(2,binning)
    ngpus = 8
    # initial guess
    u = np.zeros([nz, n, n], dtype='float32')
    psi = data.copy()

    lamd = np.zeros([ntheta, nz, n], dtype='float32')

    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    # optical flow parameters
    pars = [0.5,0, 1024//pow(2,binning), 4, 5, 1.1, 4]

    # ADMM solver
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning), ngpus) as tslv:
        # tic()
        # ucg = tslv.cg_tomo_batch(data, u, niter, dbg=True)
        # t=toc()
        # print(t)
        # dxchange.write_tiff_stack(
        #                 ucg,  name+'/cg'+'_'+str(ntheta)+'_'+str(binning)+'/rect'+'/r', overwrite=True)        
        with dc.SolverDeform(ntheta, nz, n, ptheta) as dslv:
            rho = 0.5
            h0 = psi
            for k in range(niter):
                # registration
                tic()
                flow = dslv.registration_flow_batch(
                      psi, data, mmin, mmax, flow, pars, 40)                
                t1 = toc()
                tic()
                
                # deformation subproblem
                psi = dslv.cg_deform_gpu_batch(data, psi, flow, 4,
                                               tslv.fwd_tomo_batch(u)+lamd/rho, rho)
                t2 = toc()
                # tomo subproblem
                tic()
                u = tslv.cg_tomo_batch(psi-lamd/rho, u, 4)

                t3 = toc()
                h = tslv.fwd_tomo_batch(u)
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
                    print("%d %d %.2e %.2f %.4e %.4e %.4e %.4e %d %d %d" % (k, pars[2], np.linalg.norm(
                        flow), rho, *lagr, t1, t2, t3))
                    dxchange.write_tiff_stack(
                        u,  name+'/pfw_'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    # dxchange.write_tiff_stack(
                    #   psi.real, '/local/data/viktor/brain_rec/ccc'+str(binning)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                if(pars[2] >= 20):
                    pars[2] -= 8//pow(2,binning)
                sys.stdout.flush()
                gc.collect()
# for k in range(ntheta):
#     plt.imshow(dc.flowvis.flow_to_color(flow[k]), cmap='gray')
#     plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mflow/'+str(k))
