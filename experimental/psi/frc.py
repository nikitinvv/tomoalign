import numpy as np
import sys
import dxchange
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage as ndimage
plt.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['axes.labelsize'] = 60
plt.rcParams['axes.titlesize'] = 32

def halfbit3d(data, center):
    z, y, x = np.indices((data.shape))
    r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    r = r.astype(np.int)

    nr = np.bincount(r.ravel())
    return (0.2071+1.9102/np.sqrt(nr))/(1.2071+0.9102/np.sqrt(nr)) 

def radial_profile3d(data, center):
    z, y, x = np.indices((data.shape))
    r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    r = r.astype(np.int)

    tbinre = np.bincount(r.ravel(), data.real.ravel())
    tbinim = np.bincount(r.ravel(), data.imag.ravel())
    
    nr = np.bincount(r.ravel())
    radialprofile = (tbinre+1j*tbinim) / np.sqrt(nr)
    
    return radialprofile 
def halfbit(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    nr = np.bincount(r.ravel())
    return (0.2071+1.9102/np.sqrt(nr))/(1.2071+0.9102/np.sqrt(nr)) 

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    tbinre = np.bincount(r.ravel(), data.real.ravel())
    tbinim = np.bincount(r.ravel(), data.imag.ravel())
    
    nr = np.bincount(r.ravel())
    radialprofile = (tbinre+1j*tbinim) / np.sqrt(nr)
    
    return radialprofile 


wsize = 272
fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/d/results_admm0/u/r_00000.tiff'
fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/d/results_admm1/u/r_00000.tiff'
# fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/cga_resolution1__1440/rect3/r_00000.tiff'
# fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/cga_resolution2__1440/rect3/r_00000.tiff'


f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,544))[:]
f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,544))[:]
f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]
print(f1.shape)

ff1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f1)))
ff2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f2)))

frc1 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
# print(np.min(frc1[:wsize].real))
# exit()
# frc1 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    # np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
np.save('frc1.npy',frc1)

fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/d/results_cgp1/u/r_00000.tiff'
fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/d/results_cgp2/u/r_00000.tiff'


f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,544))
f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,544))
f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]

ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

frc2 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
np.save('frc2.npy',frc2)

fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/TIFF_beta0_ram-lak_freqscl_1.00_nonrigid_SART/tomo_delta__ram-lak_freqscl_1.00_0001.tif'
fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/TIFF_beta1_ram-lak_freqscl_1.00_nonrigid_SART/tomo_delta__ram-lak_freqscl_1.00_0001.tif'


f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(1,545))
f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(1,545))
f1 = f1[f1.shape[0]//2-wsize:f1.shape[0]//2+wsize,f1.shape[1]//2-wsize:f1.shape[1]//2+wsize,f1.shape[2]//2-wsize:f1.shape[2]//2+wsize]
f2 = f2[f2.shape[0]//2-wsize:f2.shape[0]//2+wsize,f2.shape[1]//2-wsize:f2.shape[1]//2+wsize,f2.shape[2]//2-wsize:f2.shape[2]//2+wsize]


ff1 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f1),workers=-1))
ff2 = sp.fft.fftshift(sp.fft.fftn(sp.fft.fftshift(f2),workers=-1))

frc3 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
np.save('frc3.npy',frc3)
# 

ff1 = np.zeros([2*wsize,2*wsize,2*wsize],dtype='complex64')
hbit = halfbit3d(ff1,np.array([2*wsize,2*wsize,2*wsize])//2)
# hbit[wsize//2:]=hbit[wsize//2]
np.save('hbit.npy',hbit)

# exit()
# # 
frc1=np.load('frc1.npy')
# print(frc1.shape)
frc2=np.load('frc2.npy')
frc3=np.load('frc3.npy')

hbit=np.load('hbit.npy')

# frc1 = ndimage.zoom(frc1.real,8,order=2)
# frc1+=(np.random.random(frc1.real.shape)-0.5)*0.01
# frc2 = ndimage.zoom(frc2.real,8,order=2)
# frc2+=(np.random.random(frc2.real.shape)-0.5)*0.01

# frc3 = ndimage.zoom(frc3.real,8,order=2)
# frc3+=(np.random.random(frc3.real.shape)-0.5)*0.1

# exit()

# hbit=np.load('hbit.npy').real
# hbit = ndimage.zoom(hbit.real,2,order=1)

plt.figure(figsize=(7,4))
plt.plot(frc3[:wsize].real,linewidth=1.5, label=r'pCG, -')

plt.plot(frc2[:wsize].real,linewidth=1.5, label=r'OF non-dense, 195nm')
plt.plot(frc1[:wsize].real,linewidth=1.5, label=r'OF dense, 160nm')

plt.plot(hbit[:wsize],linewidth=1.5,label=r'1/2-bit')

plt.grid()
# plt.xlim([0,168])
plt.ylim([0,1])
lg = plt.legend(loc="upper right",fontsize=16, title=r'Method, resolution')
lg.get_title().set_fontsize(16)
plt.xticks(np.arange(0,168,33),[0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
plt.yticks(np.arange(0,1.1,0.2),[0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
# plt.xticks(np.arange(0,0.2,1.1),fontsize=16)
# plt.arrow(200, 0.9, -127, -0.65, color='gray',width=0.02, head_width=0) 
# plt.arrow(200, 0.7, -72, -0.48, color='gray',width=0.02, head_width=0) 


plt.ylabel('FSC',rotation=90, fontsize = 16)
axes1 = plt.gca()
axes2 = axes1.twiny()
axes1.set_xlabel('Spatial/Nyquist frequency', fontsize=16)
axes2.set_xlabel('Spatial resolution (nm)', fontsize=16)
axes2.set_xticks(np.arange(0.2, 1.1, 0.2))

axes2.set_xticklabels(np.int32(44/np.arange(0.2,1.1,0.2)),fontsize=14)


plt.tight_layout()

plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/frc_revision_'+str(wsize)+'.png',dpi=300)
# dxchange.write_tiff(f1,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f1',overwrite=True)
# dxchange.write_tiff(f2,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f2',overwrite=True)
