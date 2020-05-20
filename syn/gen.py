import matplotlib.pyplot as plt 
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import elasticdeform
import tomocg as tc
import deformcg as dc

def deform_data(data):
    points = [3, 3]
    sigma = 8
    for k in range(data.shape[0]):    
       displacement = (np.random.rand(2, *points) - 0.5) * sigma 
       data[k] = elasticdeform.deform_grid(data[k], displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None) 
    return data

def cyl(r,c,h,rx,ry,rz,n):
    [x,y] = np.meshgrid(np.arange(-n//2,n//2),np.arange(-n//2,n//2))
    x=x/n*2
    y=y/n*2
    circ1 = ((x-c[1])**2+(y-c[0])**2<r).astype('float32')
    circ2 = ((x-c[1])**2+(y-c[0])**2<r-1/512).astype('float32')
    f = np.zeros([n,n,n],dtype='float32')
    f[n//2-h//2:n//2+h//2]=circ1-circ2
    print(np.linalg.norm(f))
    f=ndimage.rotate(f,rx,axes=(1,0),reshape=False,order=1)
    f=ndimage.rotate(f,ry,axes=(1,2),reshape=False,order=1)
    f=ndimage.rotate(f,rz,axes=(2,0),reshape=False,order=1)
    
    return f

# f = cyl(0.01,[0.1,0.2],256,30,45,30,256)
# f = f+cyl(0.01,[-0.1,-0.2],256,-30,-15,30,256)
# f = f+cyl(0.01,[-0.3,-0.3],256,-30,-95,-40,256)
# f = f+cyl(0.01,[-0.4,0.4],256,15,30,90,256)
# f = f+cyl(0.01,[0.4,-0.45],256,90,30,90,256)
# f = f+cyl(0.01,[0.2,-0.25],256,-90,30,15,256)
# f = f+cyl(0.01,[0.3,-0.15],256,-10,110,-15,256)
# f[f>1]=1
# dxchange.write_tiff(f,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/f',overwrite=True)
f=np.array(dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/f.tiff').astype('float32'),order='C')

ntheta = 16
theta=np.array(np.linspace(0,np.pi,ntheta,endpoint=False),order='C')
nz,n = f.shape[0:2]
center=n/2
pnz=256
print(nz,n)
print(theta)
with tc.SolverTomo(theta, ntheta, nz, n, pnz, center, 1) as tslv:
    with dc.SolverDeform(ntheta, nz, n, 16) as dslv:
        data=tslv.fwd_tomo_batch(f)
        #ddata = deform_data(data)
        dxchange.write_tiff(data,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/data',overwrite=True)