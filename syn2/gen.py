import matplotlib.pyplot as plt 
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import elasticdeform
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib
import gc
import concurrent.futures as cf
import threading
from multiprocessing import Pool
from scipy import ndimage
from itertools import repeat
from functools import partial
def genang(numproj,nProj_per_rot):
    prime = 3
    pst = 0
    pend = 360
    seq = []
    i = 0
    sgn = 1 # for switching direction
    while len(seq) < numproj:
        b = i
        i += 1
        r = 0
        q = 1 / prime
        while (b != 0):
            a = np.mod(b, prime)
            r += (a * q)
            q /= prime
            b = np.floor(b / prime)
        r *= ((pend-pst) / nProj_per_rot)
        k = 0
        while (np.logical_and(len(seq) < numproj, k < nProj_per_rot)):
            if(sgn==1):
                seq.append(pst+ (r + k * (pend-pst) / nProj_per_rot))
            else:
                seq.append(pend-((1-r) + k * (pend-pst) / nProj_per_rot))
            k += 1
        #sgn*=-1
    return seq


def myplot(u, psi, flow, binning,alpha=0):
    [ntheta, nz, n] = psi.shape

    plt.figure(figsize=(10, 7))
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
    if not os.path.exists('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/flowfw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)):
        os.makedirs(
            '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/flowfw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha))
    plt.savefig(
        '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/flowfw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/'+str(k))
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
    flownew = ndiage.zoom(flow,(1,2,2,1),order=1)/2    
    return unew,psinew,lamdnew,flownew
    
    
def deform_data(u,theta,displacement0,k):
 
    displacement = displacement0*(1-np.exp(np.linspace(0,1,ntheta)[k]))
    ud = elasticdeform.deform_grid(u, displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None) 
    print(np.linalg.norm(ud))
    with tc.SolverTomo(theta[k:k+1], 1, nz, ne, 64, n/2+(ne-n)/2, 4) as tslv:
        data=tslv.fwd_tomo_batch(ud)
    return data

def deform_data_batch(u,theta,displacement0):
    res=np.zeros([len(theta),nz,ne],dtype='float32')  
    with cf.ThreadPoolExecutor(32) as e:
        shift = 0
        for res0 in e.map(partial(deform_data, u, theta, displacement0), range(0, len(theta))):
            res[shift] = res0
            shift += 1
    return res

def cyl(r,c,h,rx,ry,rz,n):
    [x,y] = np.meshgrid(np.arange(-n//2,n//2),np.arange(-n//2,n//2))
    x=x/n*2
    y=y/n*2
    circ1 = ((x-c[1])**2+(y-c[0])**2<r).astype('float32')
    circ2 = ((x-c[1])**2+(y-c[0])**2<r-0.9/n).astype('float32')
    f = np.zeros([n,n,n],dtype='float32')
    f[n//2-h//2:n//2+h//2]=circ1-circ2
    print(np.linalg.norm(f))
    f=ndimage.rotate(f,rx,axes=(1,0),reshape=False,order=3)
    f=ndimage.rotate(f,ry,axes=(1,2),reshape=False,order=3)
    f=ndimage.rotate(f,rz,axes=(2,0),reshape=False,order=3)
    
    return f

f = cyl(0.01,[0.1,0.2],256,30,45,30,256)
f = f+0.4*cyl(0.01,[-0.1,-0.2],256,-30,-15,30,256)
f = f+0.6*cyl(0.01,[-0.3,-0.3],256,-30,-95,-40,256)
f = f+cyl(0.01,[-0.4,0.4],256,15,30,90,256)
f = f+0.8*cyl(0.01,[0.4,-0.45],256,90,30,90,256)
f = f+0.6*cyl(0.01,[0.2,-0.25],256,-90,30,15,256)
f = f+0.4*cyl(0.01,[0.3,-0.15],256,-10,110,-15,256)
f[f>1]=1

n=256
nz=256
ne=3*n//2
fe = np.zeros([nz,ne,ne],dtype='float32')

#dxchange.write_tiff(f,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/f256',overwrite=True)
#r = np.random.random(fe.shape)
#np.save('rr',r)

r = np.load('rr.npy')
fe = (ndimage.filters.gaussian_filter(r, 3, truncate=8).astype('float32')-0.5)*12
[x,y]=np.mgrid[-ne//2:ne//2,-ne//2:ne//2]
circ = (2*x/ne)**2+(2*y/ne)**2<1
fe[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2] += f
f = (fe-np.min(fe))/(np.max(f)-np.min(fe))*circ

dxchange.write_tiff(f,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fb256',overwrite=True)

ntheta = 384

binning=0

# points = [3, 3, 3]
# r=np.random.rand(3, *points)
# np.save('r',r)#
alpha=5e-4
r = np.load('r.npy')
for irot in [384]: 
    print('irot',irot)
    theta = np.array(genang(ntheta,irot)).astype('float32')/360*np.pi*2
    for idef in [6,10,14,18,22]:
        print('idef',idef)
        data0 = unpad(deform_data_batch(f,theta,r*idef),ne,n)
        with tc.SolverTomo(theta, ntheta, nz, ne, 256, n/2+(ne-n)/2, 1) as tslv:
            with dc.SolverDeform(ntheta, nz, n, 16) as dslv:
                for inoise in range(0,6):
                    print('inoise',inoise)
                    if(inoise==0):
                        data=data0
                    else:
                        data = np.random.poisson(data0*128/pow(2,inoise)).astype('float32')*pow(2,inoise)/128

                    dxchange.write_tiff(data,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/datad/data_'+str(idef)+'_'+str(inoise)+'_'+str(irot),overwrite=True)
                    data-=np.mean(data)
                    

                    print('cg')                                        
                    u = np.zeros([nz, ne, ne], dtype='float32')
                    ucg = tslv.cg_tomo_batch(pad(data,ne,n), u, 64) 
                    dxchange.write_tiff(ucg[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ucg'+str(idef)+'_'+str(inoise)+'_'+str(irot),overwrite=True)
                   # continue
                    

                    print('of')                                        
                    mmin,mmax = find_min_max(data)
                    u = np.zeros([nz, ne, ne], dtype='float32')
                    psi = data.copy()
                    lamd = np.zeros([ntheta, nz, n], dtype='float32')
                    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
                    
                    # optical flow parameters
                    pars = [0.5,1, 256, 8, 5, 1.1, 4]
                    rho = 0.5
                    h0 = psi
                    for k in range(256):
                        flow = dslv.registration_flow_batch(
                            psi, data, mmin, mmax, flow.copy(), pars, 16) 
                        psi = dslv.cg_deform_gpu_batch(data, psi, flow, 4,
                                                    unpad(tslv.fwd_tomo_batch(u),ne,n)+lamd/rho, rho)
                        u = tslv.cg_tomo_batch(pad(psi-lamd/rho,ne,n), u, 4)                    
                        h = unpad(tslv.fwd_tomo_batch(u),ne,n)
                        lamd = lamd+rho*(h-psi)

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
                            sys.stdout.flush()           
                            dxchange.write_tiff_stack(
                                u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'/rect'+str(k)+'/r', overwrite=True)
                            dxchange.write_tiff_stack(
                            psi.real, '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'/psi'+str(k)+'/r', overwrite=True)
                            if not os.path.exists('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'/flownpy'):
                                    os.makedirs('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'/flownpy')
                            np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'/flownpy/'+str(k),flow)

                        # Updates
                        rho = update_penalty(psi, h, h0, rho)
                        h0 = h
                        if(pars[2]>28):
                            pars[2] -= 1        


                    print('of reg')                    
                    u = np.zeros([nz, ne, ne], dtype='float32')
                    psi = data.copy()
                    psi1 = np.zeros([3,nz, ne, ne], dtype='float32')
                    lamd = np.zeros([ntheta, nz, n], dtype='float32')
                    lamd1 = np.zeros([3,nz, ne, ne], dtype='float32')                    
                    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
                    
                    # optical flow parameters
                    pars = [0.5,1, 256, 8, 5, 1.1, 4]
                    rho = 0.5
                    rho1 = 0.5
                    h0 = psi
                    h01 = psi1
                    for k in range(256):
                        # registration
                        flow = dslv.registration_flow_batch(
                            psi, data, mmin, mmax, flow.copy(), pars, 16) 
                            
                        # deformation subproblem
                        psi = dslv.cg_deform_gpu_batch(data, psi, flow, 4,
                                                    unpad(tslv.fwd_tomo_batch(u),ne,n)+lamd/rho, rho)

                        psi1 = tslv.solve_reg(u,lamd1,rho1,alpha)    
                        # tomo subproblem
                        u = tslv.cg_tomo_reg_batch(pad(psi-lamd/rho,ne,n), u, 4, rho1/rho, psi1-lamd1/rho1)                    
                        h = unpad(tslv.fwd_tomo_batch(u),ne,n)
                        h1 = tslv.fwd_reg(u)
                        
                        # lambda update
                        lamd = lamd+rho*(h-psi)
                        lamd1 = lamd1+rho1*(h1-psi1)

                        myplot(u, psi, flow, binning, alpha)
                        if(np.mod(k,4)==0):  # check Lagrangian
                            Tpsi = dslv.apply_flow_gpu_batch(psi, flow)
                            lagr = np.zeros(4)
                            lagr[0] = 0.5*np.linalg.norm(Tpsi-data)**2
                            lagr[1] = np.sum(np.real(np.conj(lamd)*(h-psi)))
                            lagr[2] = rho/2*np.linalg.norm(h-psi)**2
                            lagr[3] = np.sum(lagr[0:3])
                            print("%d %d %.2e %.2f %.4e %.4e %.4e %.4e " % (k, pars[2], np.linalg.norm(
                                flow), rho, *lagr))
                            sys.stdout.flush()           
                            dxchange.write_tiff_stack(
                                u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/rect'+str(k)+'/r', overwrite=True)
                            dxchange.write_tiff_stack(
                            psi.real, '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/psi'+str(k)+'/r', overwrite=True)
                            if not os.path.exists('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/flownpy'):
                                    os.makedirs('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/flownpy')
                            np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/flownpy/'+str(k),flow)


                        # Updates
                        rho = update_penalty(psi, h, h0, rho)
                        rho1 = update_penalty(psi1, h1, h01, rho1)
                        h0 = h
                        h01 = h1
                        if(pars[2]>28):
                            pars[2] -= 1                           
                        

                    
                    print('cg reg')
                    u = np.zeros([nz, ne, ne], dtype='float32')
                    psi = data.copy()
                    psi1 = np.zeros([3,nz, ne, ne], dtype='float32')
                    lamd = np.zeros([ntheta, nz, n], dtype='float32')
                    lamd1 = np.zeros([3,nz, ne, ne], dtype='float32')
                    
                    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
                    
                    # optical flow parameters
                    rho1 = 0.5
                    h01 = psi1
                    for k in range(256):
                        psi = data
                        psi1 = tslv.solve_reg(u,lamd1,rho1,alpha)    
                        # tomo subproblem
                        u = tslv.cg_tomo_reg_batch(pad(psi,ne,n), u, 4, rho1, psi1-lamd1/rho1)                    
                        h = unpad(tslv.fwd_tomo_batch(u),ne,n)
                        h1 = tslv.fwd_reg(u)
                        
                        # lambda update
                        lamd1 = lamd1+rho1*(h1-psi1)

                        # checking intermediate results
                        if(np.mod(k,4)==0):  # check Lagrangian
                            lagr = np.zeros(4)
                            lagr[2] = 1/2*np.linalg.norm(h-psi)**2
                            lagr[3] = np.sum(lagr[0:3])
                            print("%d %.2e %.2f %.4e %.4e %.4e %.4e " % (k,  np.linalg.norm(
                                flow), rho1, *lagr))
                            sys.stdout.flush()           
                            dxchange.write_tiff_stack(
                                u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/cg_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/rect'+str(k)+'/r', overwrite=True)    
                        # Updates
                        rho1 = update_penalty(psi1, h1, h01, rho1)
                        h01 = h1
                        