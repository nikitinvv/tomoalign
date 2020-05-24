import dxchange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#plt.rcParams['xtick.major.size'] = 16
plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
#matplotlib.rcParams['text.latex.preamble'] = [r'\bold']
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.rc('font', family='serif',fontweight='bold')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 16



data = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/data.tiff')
datad = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/datad.tiff')
ntheta=384
f = np.zeros(ntheta)
f[ntheta//4:ntheta]=pow(np.sin(np.linspace(0,np.pi/2,ntheta*3//4)),1/3)
#f[ntheta//2:ntheta]=0.8+np.arange(0,ntheta/2)/ntheta/4
plt.figure(figsize=(10,3))
plt.plot(f,linewidth=3)
plt.ylabel(r'\textbf{deformation}')
plt.xlabel(r'\textbf{proj id (angle)}')
plt.ylim([0,1])
plt.xlim([0,ntheta+1])
plt.xticks(np.arange(0,385,96),[r'\textbf{0(0)}',r'\textbf{96$(\pi)$}',r'\textbf{192$(2\pi)$}',r'\textbf{288$(3\pi)$}',r'\textbf{384$(4\pi)$}'],fontsize=15)
plt.yticks(np.arange(0,1.1,0.2),[r'\textbf{0}',r'\textbf{0.2}',r'\textbf{0.4}',r'\textbf{0.6}',r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=15)
plt.text(124,0.43,r'\textbf{fast deformation}',fontsize=18)
plt.text(247,0.81,r'\textbf{slow deformation}',fontsize=18)
plt.text(2,0.03,r'\textbf{no deformation}',fontsize=18)

plt.plot(104,f[104],'ro',markersize=10)
plt.plot(104+96,f[104+96],'go',markersize=10)
plt.plot(104+96*2,f[104+96*2],'bo',markersize=10)

plt.yticks(fontsize=14)
plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/plt.png',bbox_inches = 'tight')  
plt.show()

for k in range(104,384,96):
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/dif'+str(k)+'.png',datad[k]-data[k],vmin=-20,vmax=20,cmap='gray')
    plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/proj'+str(k)+'.png',data[k],vmin=0,vmax=40,cmap='gray')
  #  plt.show()
   # plt.savefig(,bbox_inches = 'tight')  
plt.figure(figsize=(1,8))
img = plt.imshow(np.array([[-1,1]]), cmap="gray")
plt.gca().set_visible(False)
#cax = plt.axes([0.1, 0.3, 0.2, 0.2])
cb=plt.colorbar(orientation="vertical",ticks=[-1, 0, 1])
cb.ax.tick_params(labelsize=18)
cb.ax.set_yticklabels([r'\textbf{-20}', r'\textbf{0}', r'\textbf{20}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/colorbar.png",bbox_inches = 'tight')
plt.figure(figsize=(1,8))
img = plt.imshow(np.array([[-1,1]]), cmap="gray")
plt.gca().set_visible(False)
#cax = plt.axes([0.1, 0.3, 0.2, 0.2])
cb=plt.colorbar(orientation="vertical",ticks=[-1, 0, 1])
cb.ax.tick_params(labelsize=18)
cb.ax.set_yticklabels([r'\textbf{0}', r'\textbf{20}', r'\textbf{40}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/colorbarproj.png",bbox_inches = 'tight')

u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/fw__384/rect236/r_00000.tiff',ind=np.arange(0,256))
ucg = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucg.tiff')
f = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/f256.tiff')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/uz.png',u[u.shape[0]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/uy.png',u[:,u.shape[1]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ux.png',u[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgz.png',ucg[u.shape[0]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgy.png',ucg[:,u.shape[1]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgx.png',ucg[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/fz.png',f[f.shape[0]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/fy.png',f[:,f.shape[1]//2],vmin=0,vmax=1,cmap='gray')
plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/fx.png',f[:,:,f.shape[2]//2],vmin=0,vmax=1,cmap='gray')

#data = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/datad.tiff')
psi = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/fw__384/psi236/r_00000.tiff',ind=np.arange(0,384))
flow = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/fw__384/flownpy/236.npy')

import deformcg as dc
with dc.SolverDeform(384, 256, 256, 16) as dslv:
    Tpsi = dslv.apply_flow_gpu_batch(psi, flow)
    for k in range(104,384,96):
        plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/flow'+str(k)+'.png',dc.flowvis.flow_to_color(flow[k]))
        plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/Tpsi'+str(k)+'.png',Tpsi[k],vmin=0,vmax=40,cmap='gray')
        diff=datad[k]-Tpsi[k]
        diff[0,0]=20
        diff[0,1]=-1
        plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/difTpsi'+str(k)+'.png',diff,vmin=-20,vmax=20,cmap='gray')
    
