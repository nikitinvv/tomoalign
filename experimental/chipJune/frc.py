import numpy as np
import sys
import dxchange
import matplotlib
import matplotlib.pyplot as plt
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

fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098/fw2000_0rotated3_20_0/r_00000.tiff'
fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098/fw2000_1rotated3_20_0/r_00000.tiff'
w = 256
f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,400))
f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,400))
print(f1.shape)
print(f2.shape)
f2 = f2[f1.shape[0]//2-w//2:f1.shape[0]//2+w//2,f1.shape[1]//2-w//2:f1.shape[1]//2+w//2,f1.shape[2]//2-w//2:f1.shape[2]//2+w//2]
f1 = f1[f1.shape[0]//2-w//2:f1.shape[0]//2+w//2,f1.shape[1]//2-w//2:f1.shape[1]//2+w//2,f1.shape[2]//2-w//2:f1.shape[2]//2+w//2]
dxchange.write_tiff_stack(f1,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098/t1/t',overwrite=True)
dxchange.write_tiff_stack(f2,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098/t2/t',overwrite=True)

exit()

print(np.linalg.norm(f1))
print(np.linalg.norm(f2))
ff1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f1)))
ff2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f2)))


frc1 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))
print(frc1)

hbit = halfbit3d(ff1,np.array(ff1.shape)//2)


# np.save('frc1.npy',frc1)
# np.save('hbit.npy',hbit)
# exit()

frc1=np.load('frc1.npy')
hbit=np.load('hbit.npy')

plt.figure(figsize=(7,4))

plt.plot(frc1[:w].real,linewidth=1.5, label=r'OF')

plt.plot(hbit[:w],linewidth=1.5,label=r'1/2-bit')

plt.grid()
plt.xlim([0,w/2])
plt.ylim([0,1])
lg = plt.legend(loc="upper right",fontsize=16, title=r'Method, resolution')
lg.get_title().set_fontsize(16)
plt.xticks(np.arange(0,256,51),[0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
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
plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipJune/chip_16nmZP_tube_lens_interlaced_2000prj_3s_098/frc.png',dpi=300)
# dxchange.write_tiff(f1,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f1',overwrite=True)
# dxchange.write_tiff(f2,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f2',overwrite=True)
