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
#import tomopy 
# matplotlib.use('Agg')
# def myplot(u, psi, flow, binning):
#     ids = np.argsort(np.linalg.norm(flow,axis=(1,2,3)))[::-1]
#     [ntheta, nz, n] = psi.shape

#     plt.figure(figsize=(20, 14))
#     for k in range(4):
#         plt.subplot(2, 4, k+1)
#         tit = 'y:',flow[ids[k],0],'x:',flow[ids[k],1]
#         plt.title(tit)
#         plt.imshow(psi[ids[k]], cmap='gray')

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
    if not os.path.exists('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cflowfw_'+str(binning)+'_'+str(ntheta)):
        os.makedirs(
            '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cflowfw_'+str(binning)+'_'+str(ntheta))
    plt.savefig(
        '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cflowfw_'+str(binning)+'_'+str(ntheta)+'/'+str(k))
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
# import tomopy
# prj=np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cprjbin0p2.npy')
# print(prj.shape)
# prj = tomopy.downsample(prj, level=1)
# prj = tomopy.downsample(prj, level=1, axis=1)
# print(np.linalg.norm(prj[-1]))
# exit()
# np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cprjbin1p2',prj)        
# prj = tomopy.downsample(prj, level=1)
# prj = tomopy.downsample(prj, level=1, axis=1)
# np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cprjbin2p2',prj)        
def find_min_max(data):
    s = np.std(data,axis=(1,2))    
    m = np.mean(data,axis=(1,2))
    mmin = m-2*s
    mmax = m+2*s
    return mmin,mmax

if __name__ == "__main__":
    binning = 2
    data = np.zeros([4000,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    
    data[:,:data.shape[1]//2] = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin0p1new.npy').astype('float32')[::4000//data.shape[0],::2048//data.shape[1],::2448//data.shape[2]]                               
    data[:,data.shape[1]//2:] = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin0p2new.npy').astype('float32')[::4000//data.shape[0],::2048//data.shape[1],::2448//data.shape[2]]                                   
    #prj[:,:prj.shape[1]//2] = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin1p1.npy').astype('float32')[::4000//prj.shape[0]]#::1024//prj.shape[1],::2448//prj.shape[2]]                               
    #prj[:,prj.shape[1]//2:] = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin1p2.npy').astype('float32')[::4000//prj.shape[0]]#,::1024//prj.shape[1],::2448//prj.shape[2]]                                   
    
    
    data[np.isnan(data)]=0
    data = data[:,768//pow(2,binning):-768//pow(2,binning),:]
    dxchange.write_tiff_stack(data,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cdatabin0/r')
    theta = np.array(np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/theta.npy').astype('float32')[::4000//data.shape[0]],order='C')
    # data
    
    center = np.int(sys.argv[1])
    #data = data[:,:,:center*2//pow(2,binning)]
    data = data[:,:,456//pow(2,binning):-456//pow(2,binning)]
    center-=456
    #print(theta)
    print('shape',data.shape,'center',center//pow(2,binning))
    #center-=122*4//pow(2,binning)

    [ntheta, nz, n] = data.shape  # object size n x,y
    data-=np.mean(data)
    mmin,mmax = find_min_max(data)
    
    # exit()
    niter = 32  # tomography iterations
    pnz = 32  # number of slice partitions for simultaneous processing in tomography
    ptheta = 400
    ngpus=4
    # initial guess
    u = np.zeros([nz, n, n], dtype='float32')
    psi = data.copy()

    lamd = np.zeros([ntheta, nz, n], dtype='float32')

    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    # optical flow parameters
    pars = [0.5,0, 128, 4, 5, 1.1, 4]

   # print(np.linalg.norm(data))
    # ADMM solver
    with tc.SolverTomo(theta, ntheta, nz, n, pnz, center/pow(2, binning), ngpus) as tslv:
        # ucg = tslv.cg_tomo_batch(data, u, niter)
        # dxchange.write_tiff_stack(
        #                 ucg,  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cg'+'_'+str(ntheta)+'/rect'+'/r', overwrite=True)
        # exit()
        with dc.SolverDeform(ntheta, nz, n, ptheta) as dslv:
            rho = 0.5
            h0 = psi
            for k in range(niter):
                # registration
                tic()
                #print('b',np.linalg.norm(dslv.apply_flow_gpu_batch(psi,flow)-data))
                flow = dslv.registration_flow_batch(
                      psi, data, mmin, mmax, flow, pars, 40)
                #print('a',np.linalg.norm(dslv.apply_flow_gpu_batch(psi,flow)-data))
 
                t1 = toc()
                tic()
                
                # deformation subproblem
                psi = dslv.cg_deform_gpu_batch(data, psi, flow, 4,
                                               tslv.fwd_tomo_batch(u)+lamd/rho, rho)
                # psi1 = dslv.cg_deform(data, psi, flow, 4,
                #                     tslv.fwd_tomo_batch(u)+lamd/rho, rho, 40)
                # print(np.linalg.norm(psi1-psi2))
                # psi=psi2
                t2 = toc()

                # tomo subproblem
                tic()
                u = tslv.cg_tomo_batch(psi-lamd/rho, u, 4)

                t3 = toc()
                h = tslv.fwd_tomo_batch(u)
                # lambda update
                lamd = lamd+rho*(h-psi)

                # checking intermediate results
                #myplot(u, psi, flow, binning)
                if(k==31):  # check Lagrangian
                    Tpsi = dslv.apply_flow_gpu_batch(psi, flow)
                    lagr = np.zeros(4)
                    lagr[0] = 0.5*np.linalg.norm(Tpsi-data)**2
                    lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                    lagr[2] = rho/2*np.linalg.norm(h-psi)**2
                    lagr[3] = np.sum(lagr[0:3])
                    print(k, pars[2], np.linalg.norm(
                        flow), rho, lagr, t1, t2, t3)
                    dxchange.write_tiff_stack(
                        u,  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/cent'+str(center)+'fw_'+str(binning)+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                    # dxchange.write_tiff_stack(
                    #   psi.real, '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/ccc'+str(binning)+'_'+str(ntheta)+'/psir'+str(k)+'/r',  overwrite=True)

                # Updates
                rho = update_penalty(psi, h, h0, rho)
                h0 = h
                if(pars[2] >= 12):
                    pars[2] -= 2
                sys.stdout.flush()
                gc.collect()
# for k in range(ntheta):
#     plt.imshow(dc.flowvis.flow_to_color(flow[k]), cmap='gray')
#     plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mflow/'+str(k))
