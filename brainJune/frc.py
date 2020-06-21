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

fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/cga_resolution1__1440/rect3/r_00000.tiff'
fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/cga_resolution2__1440/rect3/r_00000.tiff'

f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,1024))#[0,100:-100,100:-100]
f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,1024))#[0,100:-100,100:-100]

ff1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f1)))
ff2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f2)))

frc1 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))

fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/fw_resolution1_1440/rect24/r_00000.tiff'
fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/fw_resolution2_1440/rect24/r_00000.tiff'

f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,1024))#[0,100:-100,100:-100]
f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,1024))#[0,100:-100,100:-100]

ff1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f1)))
ff2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f2)))

frc2 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))

fname1 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/shiftfw_resolution1_1440/rect12/r_00000.tiff'
fname2 = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/shiftfw_resolution2_1440/rect12/r_00000.tiff'
f1 = dxchange.read_tiff_stack(fname1, ind = np.arange(0,1024))#[0,100:-100,100:-100]
f2 = dxchange.read_tiff_stack(fname2, ind = np.arange(0,1024))#[0,100:-100,100:-100]

ff1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f1)))
ff2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f2)))

frc3 = radial_profile3d(ff1*np.conj(ff2),np.array(ff1.shape)//2)/\
    np.sqrt(radial_profile3d(np.abs(ff1)**2,np.array(ff1.shape)//2)*radial_profile3d(np.abs(ff2)**2,np.array(ff1.shape)//2))




hbit = halfbit3d(ff1,np.array(ff1.shape)//2)



plt.figure(figsize=(7,4))

plt.plot(frc1[:f1.shape[1]//2].real,linewidth=2, label='pCG')
plt.plot(frc3[:f1.shape[1]//2].real,linewidth=2, label='OF non-dense')
plt.plot(frc2[:f1.shape[1]//2].real,linewidth=2, label='OF dense')

plt.plot(hbit[:f1.shape[1]//2],linewidth=2,label='1/2-bit')

plt.grid()
plt.xlim([0,513])
plt.ylim([0,1])
plt.legend(loc="upper right",fontsize=22)
plt.xticks(np.arange(0,516,103),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=18)
plt.yticks(np.arange(0,1.1,0.2),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=18)
# plt.xticks(np.arange(0,0.2,1.1),fontsize=16)
# plt.arrow(200, 0.9, -127, -0.65, color='gray',width=0.02, head_width=0) 
# plt.arrow(200, 0.7, -72, -0.48, color='gray',width=0.02, head_width=0) 


plt.savefig('/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/frc.png')
# dxchange.write_tiff(f1,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f1',overwrite=True)
# dxchange.write_tiff(f2,'/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/f2',overwrite=True)
