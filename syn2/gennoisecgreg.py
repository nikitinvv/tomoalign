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


def myplot(u, psi, flow, binning):
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
    if not os.path.exists('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/'+'/noiseflowfw_'+'_'+str(ntheta)):
        os.makedirs(
            '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/'+'/noiseflowfw_'+'_'+str(ntheta))
    plt.savefig(
        '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/'+'/noiseflowfw_'+'_'+str(ntheta)+'/'+str(k))
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
    
    
def deform_data(data):
    points = [3, 3]
    sigma = 11
    displacement0 = (np.random.rand(2, *points) - 0.5) * sigma 
    ntheta=data.shape[0]
    #no deformation
    #fast deformation
    ddata=data.copy()
    for k in range(ntheta//4,ntheta):    
       displacement = displacement0*pow(np.sin(np.linspace(0,np.pi/2,ntheta*3//4)[k-ntheta//4]),1/2)
       ddata[k] = elasticdeform.deform_grid(ddata[k], displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None) 
    #slowdown
    # displacement0=displacement
    # for k in range(ntheta//2,ntheta):    
    #    displacement = displacement0+displacement0/(ntheta)*(k-ntheta/2)/8
    #    ddata[k] = elasticdeform.deform_grid(ddata[k], displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None) 
    
    return ddata

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

# f = cyl(0.01,[0.1,0.2],256,30,45,30,256)
# f = f+cyl(0.01,[-0.1,-0.2],256,-30,-15,30,256)
# f = f+cyl(0.01,[-0.3,-0.3],256,-30,-95,-40,256)
# f = f+cyl(0.01,[-0.4,0.4],256,15,30,90,256)
# f = f+cyl(0.01,[0.4,-0.45],256,90,30,90,256)
# f = f+cyl(0.01,[0.2,-0.25],256,-90,30,15,256)
# f = f+cyl(0.01,[0.3,-0.15],256,-10,110,-15,256)
# f[f>1]=1
# dxchange.write_tiff(f,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/f256',overwrite=True)
f=dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/f256.tiff').astype('float32')


nz,n = f.shape[0:2]
ntheta = 3*n//2
theta=np.linspace(0,4*np.pi,ntheta,endpoint=False).astype('float32')
theta = np.array(genang(ntheta,ntheta//2)).astype('float32')/360*np.pi*2
print(theta)

binning=0
ne=n
# with tc.SolverTomo(theta, ntheta, nz, n, nz//4, n/2, 4) as tslv:
#     ddata=tslvfwd_tomo_batch(f)
alpha = 1e-3
with tc.SolverTomo(theta, ntheta, nz, ne, 256, n/2+(ne-n)/2, 1) as tslv:
    with dc.SolverDeform(ntheta, nz, n, 16) as dslv:
        data=dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/datadnoise.tiff').copy()
        u = np.zeros([nz, ne, ne], dtype='float32')
        ucg = tslv.cg_tomo_batch(pad(data,ne,n), u, 64) 
        dxchange.write_tiff(ucg[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgnoise',overwrite=True)
        #exit()
        #optflow
        mmin,mmax = find_min_max(data)
        u = np.zeros([nz, ne, ne], dtype='float32')
        psi = data.copy()
        psi1 = np.zeros([3,nz, n, n], dtype='float32')
        lamd = np.zeros([ntheta, nz, n], dtype='float32')
        lamd1 = np.zeros([3,nz, n, n], dtype='float32')
        
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
                    u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  '/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/'+'/noisecgregfw_'+'_'+str(ntheta)+str(alpha)+'/rect'+str(k)+'/r', overwrite=True)                
            # Updates
            rho1 = update_penalty(psi1, h1, h01, rho1)
            h01 = h1
            