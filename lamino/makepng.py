import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

import gc
import scipy.ndimage as ndimage
import cv2
import deformcg as dc
plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino//rec/r_00000.tiff',ind=np.arange(0,512)).copy()
ur = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino//recr/r_00000.tiff',ind=np.arange(0,512)).copy()
#ucg = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/cg_168.tiff').copy()
ucg = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/res/rect160/r_00000.tiff',ind=np.arange(0,512)).copy()
#un = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/recn/r_00000.tiff',ind=np.arange(0,512)).copy()
un = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/res/rect256/r_00000.tiff',ind=np.arange(0,512)).copy()


#ucg = ndimage.filters.median_filter(ucg,3)

vmin=-1.3e-5
vmax=1.3e-5
a=u[290];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/uz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=u[:,u.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/uy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=u[:,:,u.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/ux.png',a,vmin=vmin,vmax=vmax,cmap='gray')

a=ur[290];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/urz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=ur[:,u.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/ury.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=ur[:,:,u.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/urx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# u=[]

a=ucg[290];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/ucgz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=ucg[:,ucg.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/ucgy.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=ucg[:,:,ucg.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/ucgx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# ucg=[]

a=un[290];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/unz.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=un[:,un.shape[1]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/uny.png',a,vmin=vmin,vmax=vmax,cmap='gray')
a=un[:,:,un.shape[2]//2];a[0]=vmin;a[1]=vmax
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/unx.png',a,vmin=vmin,vmax=vmax,cmap='gray')
# un=[]


flow = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/flow.npy')/7
flowr = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/flowr.npy')/7
flown = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/flown.npy')/7

for k in range(0,168,167):
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/flow'+str(k)+'.png',dc.flowvis.flow_to_color(flow[k]))
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/flowr'+str(k)+'.png',dc.flowvis.flow_to_color(flowr[k]))
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/flown'+str(k)+'.png',dc.flowvis.flow_to_color(flown[k]))

plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[vmin,vmax]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[vmin, (vmin+vmax)/2, vmax])
cb.ax.tick_params(labelsize=28)
cb.ax.set_xticklabels([r'\textbf{-1.3e-5}', r'\textbf{0}', r'\textbf{1.3e-5}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/hcolorbarobj.png",bbox_inches = 'tight')
plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[-0.7,0.7]]), cmap="gray")
plt.gca().set_visible(False)
cb=plt.colorbar(orientation="horizontal",ticks=[-0.7, 0.0, 0.7])
cb.ax.tick_params(labelsize=22)
cb.ax.set_xticklabels([r'\textbf{-0.7}', r'\textbf{0.0}', r'\textbf{0.7}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data//lamino/figs/hcolorbarproj.png",bbox_inches = 'tight')