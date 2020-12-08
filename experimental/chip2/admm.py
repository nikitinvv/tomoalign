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
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_3s_068': 1249-456,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073':1249-456,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_074':1249-456,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_075':1249-456,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_076':1249-456,
}
ngpus = 4
def myplot(u, psi, flow, binning,center):
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
    if not os.path.exists(name+'/ti_bin12flowfw_'+'_'+str(ntheta)):
        os.makedirs(
            name+'/ti_bin12flowfw_'+'_'+str(ntheta))
    plt.savefig(
        name+'/ti_bin12flowfw_'+'_'+str(ntheta)+'/'+str(k))
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
    unew = ndimage.zoom(u,2,order=1)
    psinew = ndimage.zoom(psi,(1,2,2),order=1)
    lamdnew = ndimage.zoom(lamd,(1,2,2),order=1)
    flownew = ndimage.zoom(flow,(1,2,2,1),order=1)/2    
    return unew,psinew,lamdnew,flownew

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    nmerged = np.int(sys.argv[3])
    name = sys.argv[4]   
    
    w = [384,200,100]
    niter = [75,45,20]
    #niter=[1,1,1]
    binnings=[2,1,0]
    # ADMM solver
    for il in range(2):
        binning = binnings[il]
        data = np.zeros([ndsets*nth*nmerged,1536//pow(2,binning),1536//pow(2,binning)],dtype='float32')
        theta = np.zeros(ndsets*nth*nmerged,dtype='float32')
        for k in range(ndsets):
            data[k*nth+ndsets*nth*0:(k+1)*nth+ndsets*nth*0] = np.load(name+'_ti_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
            theta[k*nth+ndsets*nth*0:(k+1)*nth+ndsets*nth*0] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')            
            for imerged in range(1,nmerged):
                data[k*nth+ndsets*nth*imerged:(k+1)*nth+ndsets*nth*imerged] = np.load(name[:-1]+str(int(name[-1])+imerged)+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
                theta[k*nth+ndsets*nth*imerged:(k+1)*nth+ndsets*nth*imerged] = np.load(name[:-1]+str(int(name[-1])+imerged)+'_theta'+str(k)+'.npy').astype('float32')
        
        [ntheta, nz, n] = data.shape  # object size n x,y
           # print(theta[k*nth:(k+1)*nth])
       # exit()
       # data=data[:,256//pow(2,binning):-256//pow(2,binning),256//pow(2,binning):-256//pow(2,binning)]
        data[np.isnan(data)]=0            
        data-=np.mean(data)
        mmin,mmax = find_min_max(data)
        # pad data    
        ne = 2048//pow(2,binning)    
        #ne=n
        center = 1249-456+(ne//2-n//2)*pow(2,binning)#-256#not binned!
        pnz = 8*pow(2,binning)  # number of slice partitions for simultaneous processing in tomography
        ptheta = 20
        # initial guess

        if(il==0):
            u = np.zeros([nz, ne, ne], dtype='float32')
            psi = data.copy()
            lamd = np.zeros([ntheta, nz, n], dtype='float32')
            flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
        else:            
            #print(flow[0,:4,:4,:])           
            u, psi, lamd, flow = interpdense(u,psi,lamd,flow)
        # print(flow[0,:4,:4,:])  =
            # print(np.linalg.norm(u))
            # print(np.linalg.norm(u1[::2,::2,::2]-u))
            # print(np.linalg.norm(psi1[:,::2,::2]-psi))
            # print(np.linalg.norm(lamd1[:,::2,::2]-lamd))
            # print(np.linalg.norm(flow1[:,::2,::2]-flow))
        
        # optical flow parameters
        pars = [0.5,1, w[il], 4, 5, 1.1, 4]
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
                    myplot(u, psi, flow, binning, center)
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
                            u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  name+'/ti_bin12fw_'+'_'+str(ntheta)+'/rect'+str(k)+'/r', overwrite=True)
                        #dxchange.write_tiff_stack(
                       # psi.real, name+'/bin12fw_'+'_'+str(ntheta)+'/psi'+str(k)+'/r', overwrite=True)

                    # Updates
                    rho = update_penalty(psi, h, h0, rho)
                    h0 = h
                    pars[2] -= 4
                    sys.stdout.flush()
                    gc.collect()
