import numpy as np
import cupy as cp
import dxchange
import sys
import matplotlib.pyplot as plt 

def radial_profile(data, center):
    data = cp.array(data)
    data = cp.abs(cp.fft.fftshift(cp.fft.fftn(cp.fft.fftshift(data))))
    dxchange.write_tiff(data[256].get(),'d.tiff')
    #plt.savefig('d.png')
    z, y, x = cp.indices((data.shape))
    r = cp.sqrt((x - center[2])**2 + (y - center[1])**2+(z - center[0])**2)
    r = r.astype(np.int)

    print(data[256,306,306])
    print(r[256,306,306])
    tbin = cp.bincount(r.ravel(), data.ravel())
    nr = cp.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile.get()

if __name__ == "__main__":
    fname=3*[None]
    fname[0] = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/fw__2880/rect20/r_00000.tiff'
    fname[1] = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/shiftfw__2880/rect12/r_00000.tiff'
    fname[2] = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167/cga__2880_median1x1x1/r0000.tif'
    
    for k in range(3):
        print(fname[k])
        f = dxchange.read_tiff_stack(fname[k],ind=np.arange(0,1024))[::2,::2,::2]
        pr = radial_profile(f,np.array(f.shape)//2)
        plt.plot(np.log(pr[16:]))
    
    plt.savefig('pr')
    