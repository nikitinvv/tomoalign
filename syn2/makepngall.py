import dxchange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats 
#plt.rcParams['xtick.major.size'] = 16
plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
#matplotlib.rcParams['text.latex.preamble'] = [r'\bold']
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.rc('font', family='serif',fontweight='bold')
plt.rcParams['axes.labelsize'] = 60
plt.rcParams['axes.titlesize'] = 32

plt.figure(figsize=(8,1))
img = plt.imshow(np.array([[-0.2,0.5]]), cmap="gray")
plt.gca().set_visible(False)
#cax = plt.axes([0.1, 0.3, 0.2, 0.2])


cb=plt.colorbar(orientation="horizontal",ticks=[-0.2, 0.15, 0.4])
cb.ax.tick_params(labelsize=32)
cb.ax.set_xticklabels([r'\textbf{-0.2}', r'\textbf{0.15}', r'\textbf{0.5}']) 
plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/figsall/hcolorbarall.png",bbox_inches = 'tight')
#exit()
# np.save('r',r)#
alpha=5e-4
ntheta=384
vmin=-0.05
vmax=0.5
for irot in [96,192,384]: 
    print('irot',irot)
    for idef in [10,14]:
        print('idef',idef)
        for inoise in range(0,3,2):
            print('inoise',inoise)
            #ucg = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ucg'+str(idef)+'_'+str(inoise)+'_'+str(irot)+'.tiff')
            ucgp = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ucgp'+str(idef)+'_'+str(inoise)+'_'+str(irot)+'.tiff')            
            u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'/rect200/r_00000.tiff',ind=np.arange(0,256))
            ur = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/fw_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/rect200/r_00000.tiff',ind=np.arange(0,256))
            #ucgr = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/cg_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/rect200/r_00000.tiff',ind=np.arange(0,256))
            ucgpr = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/'+'/cgp_'+'_'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'/rect200/r_00000.tiff',ind=np.arange(0,256))
            #plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/figsall/'+'ucg'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'.png',ucg[165],vmin=vmin,vmax=vmax,cmap='gray')
            plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/figsall/'+'ucgp'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'.png',ucgp[165],vmin=vmin,vmax=vmax,cmap='gray')
            plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/figsall/'+'u'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'.png',u[165],vmin=vmin,vmax=vmax,cmap='gray')
            plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/figsall/'+'ur'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'.png',ur[165],vmin=vmin,vmax=vmax,cmap='gray')
            #plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/figsall/'+'ucgr'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'.png',ucgr[165],vmin=vmin,vmax=vmax,cmap='gray')
            plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/figsall/'+'ucgpr'+str(ntheta)+str(idef)+'_'+str(inoise)+'_'+str(irot)+'_'+str(alpha)+'.png',ucgpr[165],vmin=vmin,vmax=vmax,cmap='gray')

exit()
# data = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/datad/data_0_-1_96.tiff')
# datad = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/datad/data_10_-1_96.tiff')
# ntheta=384
# f = np.zeros(ntheta)
# f=1-np.exp(-3*np.linspace(0,1,ntheta))
# #f[ntheta//2:ntheta]=0.8+np.arange(0,ntheta/2)/ntheta/4
# plt.figure(figsize=(10,10))
# plt.plot(np.linspace(0,ntheta,ntheta),f,linewidth=8)
# plt.ylabel(r'\textbf{deformation}')
# plt.xlabel(r'\textbf{proj id (angle)}')
# plt.ylim([0,1.05])
# plt.xlim([0,ntheta+1])
# plt.xticks(np.arange(0,385,192),[r'\textbf{0$(\!0\!)$}',r'\textbf{192$(\!4\pi\!)$}',r'\textbf{384$(\!8\pi\!)$}'],fontsize=54)
# plt.yticks(np.arange(0,1.1,0.2),[r'\textbf{0}',r'\textbf{0.2}',r'\textbf{0.4}',r'\textbf{0.6}',r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=54)
# #plt.text(110,0.2,r'\textbf{fast deformation}',fontsize=27)
# #plt.text(205,0.65,r'\textbf{slow deformation}',fontsize=27)
# #plt.text(2,0.03,r'\textbf{no deformation}',fontsize=25)

# plt.plot(96,f[96],'ro',markersize=32)
# plt.plot(96*2,f[96*2],'go',markersize=32)
# plt.plot(96*3,f[96*3],'bo',markersize=32)

# #plt.yticks(fontsize=50)
# #plt.gca().invert_yaxis()

# plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/plt.png',bbox_inches = 'tight')  
# plt.show()
# #exit()
# for k in range(96,384,96):
#     plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/dif'+str(k)+'.png',datad[k]-data[k],vmin=-20,vmax=20,cmap=''gray'')
#     plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/proj'+str(k)+'.png',data[k],vmin=0,vmax=40,cmap=''gray'')
#     print('rmsd datad:',k,np.linalg.norm(datad[k]-data[k]))
# print('rmsd datad:',np.linalg.norm(datad-data))
#   #  plt.show()
#    # plt.savefig(,bbox_inches = 'tight')  
# plt.figure(figsize=(1,8))
# img = plt.imshow(np.array([[-1,1]]), cmap="'gray'")
# plt.gca().set_visible(False)
# #cax = plt.axes([0.1, 0.3, 0.2, 0.2])


# cb=plt.colorbar(orientation="vertical",ticks=[-1, 0, 1])
# cb.ax.tick_params(labelsize=18)
# cb.ax.set_yticklabels([r'\textbf{-20}', r'\textbf{0}', r'\textbf{20}']) 
# plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/colorbar.png",bbox_inches = 'tight')
# plt.figure(figsize=(1,8))
# img = plt.imshow(np.array([[-1,1]]), cmap="'gray'")
# plt.gca().set_visible(False)
# #cax = plt.axes([0.1, 0.3, 0.2, 0.2])


# cb=plt.colorbar(orientation="vertical",ticks=[-1, 0, 1])
# cb.ax.tick_params(labelsize=20)
# cb.ax.set_yticklabels([r'\textbf{0}', r'\textbf{20}', r'\textbf{40}']) 
# plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/colorbarproj.png",bbox_inches = 'tight')


# plt.figure(figsize=(8,1))
# img = plt.imshow(np.array([[0,40]]), cmap="'gray'")
# plt.gca().set_visible(False)
# #cax = plt.axes([0.1, 0.3, 0.2, 0.2])


# cb=plt.colorbar(orientation="horizontal",ticks=[0, 20, 40])
# cb.ax.tick_params(labelsize=20)
# cb.ax.set_yticklabels([r'\textbf{0}', r'\textbf{20}', r'\textbf{40}']) 
# plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/hcolorbarproj.png",bbox_inches = 'tight')

# plt.figure(figsize=(8,1))
# img = plt.imshow(np.array([[-20,20]]), cmap="'gray'")
# plt.gca().set_visible(False)
# cb=plt.colorbar(orientation="horizontal",ticks=[-20, 0, 20])
# cb.ax.tick_params(labelsize=20)
# cb.ax.set_xticklabels([r'\textbf{-20}', r'\textbf{0}', r'\textbf{20}']) 
# plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/hcolorbardifproj.png",bbox_inches = 'tight')

# plt.figure(figsize=(8,1))
# img = plt.imshow(np.array([[-8,8]]), cmap="'gray'")
# plt.gca().set_visible(False)
# cb=plt.colorbar(orientation="horizontal",ticks=[-8, 0, 8])
# cb.ax.tick_params(labelsize=20)
# cb.ax.set_xticklabels([r'\textbf{-8}', r'\textbf{0}', r'\textbf{8}']) 
# plt.savefig("/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/hcolorbardifprojp.png",bbox_inches = 'tight')

# u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fw__38410_-1_96/rect212/r_00000.tiff',ind=np.arange(0,256))
# ucg = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ucg10_-1_96.tiff')
# f = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fb256.tiff')
# f=f[:,64:-64,64:-64]
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/uz.png',u[165],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/uy.png',u[:,u.shape[1]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ux.png',u[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ucgz.png',ucg[165],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ucgy.png',ucg[:,u.shape[1]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/ucgx.png',ucg[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fz.png',f[f.shape[0]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fy.png',f[:,f.shape[1]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fx.png',f[:,:,f.shape[2]//2],vmin=0,vmax=1,cmap=''gray'')

# #data = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/datad/data_0_-1_96.tiff').copy()
# psi = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fw__38410_-1_96/psi200/r_00000.tiff',ind=np.arange(0,384))
# flow = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/fw__38410_-1_96/flownpy/200.npy')
# import deformcg as dc
# with dc.SolverDeform(384, 256, 256, 16) as dslv:
#     Tpsi = dslv.apply_flow_gpu_batch(psi, flow)
#     for k in range(96,384,96):
#         plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/flow'+str(k)+'.png',dc.flowvis.flow_to_color(flow[k]))
#         plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/Tpsi'+str(k)+'.png',Tpsi[k],vmin=0,vmax=40,cmap=''gray'')
#         plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/psi'+str(k)+'.png',psi[k],vmin=0,vmax=40,cmap=''gray'')
#         diff=datad[k]-Tpsi[k]
#         print(np.linalg.norm(diff))
#        # diff=datad[k]-psi[k]
#         #print(np.linalg.norm(diff))
        
#         diff[0,0]=8
#         diff[0,1]=-8
#         plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/difTpsi'+str(k)+'.png',diff,vmin=-8,vmax=8,cmap=''gray'')
#         diff=datad[k]-psi[k]
#         print(np.linalg.norm(diff))
#        # diff=datad[k]-psi[k]
#         #print(np.linalg.norm(diff))
        
#         diff[0,0]=2
#         diff[0,1]=-2
#         plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn2/difpsi'+str(k)+'.png',diff,vmin=-20,vmax=20,cmap=''gray'')
# exit()

# from skimage.metrics import structural_similarity as ssim
# ss1 = ssim(f,f,data_range=f.max()-f.min())
# ss2 = ssim(f[8:-8,8:-8,8:-8],ucg[8:-8,8:-8,8:-8],data_range=ucg.max()-ucg.min())
# ss3 = ssim(f[8:-8,8:-8,8:-8],u[8:-8,8:-8,8:-8],data_range=u.max()-u.min())
# ss4 = ssim(Tpsi[:,8:-8,8:-8],data[:,8:-8,8:-8],data_range=40)
# for k in range(104,384,96):
#     print('ssim',k,ssim(Tpsi[k,8:-8,8:-8],data[k,8:-8,8:-8],data_range=40))
#     print('rmsd data',k,np.linalg.norm(Tpsi[k,8:-8,8:-8]-data[k,8:-8,8:-8]))
# print('rmsd data',np.linalg.norm(Tpsi[:,8:-8,8:-8]-data[:,8:-8,8:-8]))

# print('ssim:',ss1,ss2,ss3,ss4)
# print('std:',np.std(f),np.std(ucg),np.std(u))
# print('snr:',np.mean(f)/np.std(f),np.mean(ucg)/np.std(ucg),np.mean(u)/np.std(u))

# print('rmsd u:',np.linalg.norm(f[8:-8,8:-8,8:-8]-u[8:-8,8:-8,8:-8]),np.linalg.norm(f[8:-8,8:-8,8:-8]-ucg[8:-8,8:-8,8:-8]))


# ucg = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgnoise.tiff')
# u = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/noisefw__384/rect236/r_00000.tiff',ind=np.arange(0,256))
# ucgreg = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/noisecgregfw__3840.003/rect236/r_00000.tiff',ind=np.arange(0,256))
# ureg = dxchange.read_tiff_stack('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/noisefw__3840.001/rect236/r_00000.tiff',ind=np.arange(0,256))
# flownoise = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/noisefw__384//flownpy/236.npy')
# flownoisereg = np.load('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/noisefw__3840.003//flownpy/236.npy')

# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/unoisez.png',u[165],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/unoisey.png',u[:,u.shape[1]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/unoisex.png',u[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgnoisez.png',ucg[165],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgnoisey.png',ucg[:,u.shape[1]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgnoisex.png',ucg[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgregnoisez.png',ucgreg[165],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgregnoisey.png',ucgreg[:,u.shape[1]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/ucgregnoisex.png',ucgreg[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/uregnoisez.png',ureg[165],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/uregnoisey.png',ureg[:,u.shape[1]//2],vmin=0,vmax=1,cmap=''gray'')
# plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/uregnoisex.png',ureg[:,:,u.shape[2]//2],vmin=0,vmax=1,cmap=''gray'')
# ss1 = ssim(f,f,data_range=f.max()-f.min())
# ss2 = ssim(f[8:-8,8:-8,8:-8],ucg[8:-8,8:-8,8:-8],data_range=ucg.max()-ucg.min())
# ss3 = ssim(f[8:-8,8:-8,8:-8],u[8:-8,8:-8,8:-8],data_range=u.max()-u.min())
# ss4 = ssim(f[8:-8,8:-8,8:-8],ucgreg[8:-8,8:-8,8:-8],data_range=ucgreg.max()-ucgreg.min())
# ss5 = ssim(f[8:-8,8:-8,8:-8],ureg[8:-8,8:-8,8:-8],data_range=ureg.max()-ureg.min())


# print('ssim:',ss1,ss2,ss4,ss3,ss5)
# print('std:',np.std(f),np.std(ucg),np.std(ucgreg),np.std(u),np.std(ureg))
# print('snr:',np.mean(f)/np.std(f),np.mean(ucg)/np.std(ucg),np.mean(ucgreg)/np.std(ucgreg),np.mean(u)/np.std(u),np.mean(ureg)/np.std(ureg))

# print('rmsd u:',
#     np.linalg.norm(f[8:-8,8:-8,8:-8]-ucg[8:-8,8:-8,8:-8]),\
#     np.linalg.norm(f[8:-8,8:-8,8:-8]-ucgreg[8:-8,8:-8,8:-8]),\
#     np.linalg.norm(f[8:-8,8:-8,8:-8]-u[8:-8,8:-8,8:-8]),\
#     np.linalg.norm(f[8:-8,8:-8,8:-8]-ureg[8:-8,8:-8,8:-8]))
# import deformcg as dc
# with dc.SolverDeform(384, 256, 256, 16) as dslv:
#     Tpsi = dslv.apply_flow_gpu_batch(psi, flownoise)
#     for k in range(104,384,96):
#         plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/flownoise'+str(k)+'.png',dc.flowvis.flow_to_color(flow[k]))
#     Tpsi = dslv.apply_flow_gpu_batch(psi, flownoisereg)
#     for k in range(104,384,96):
#         plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/flownoisereg'+str(k)+'.png',dc.flowvis.flow_to_color(flow[k]))

# datanoise = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/datadnoise.tiff')
# for k in range(104,384,96):
#     plt.imsave('/data/staff/tomograms/vviknik/tomoalign_vincent_data/syn/projnoise'+str(k)+'.png',datanoise[k],vmin=0,vmax=40,cmap=''gray'')            