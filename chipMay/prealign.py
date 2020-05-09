import dxchange
import numpy as np
import h5py
import sys
import skimage.feature
def apply_shift(psi, p):
    """Apply shift for all projections."""
    [nz,n] = psi.shape
    tmp = np.zeros([2*nz, 2*n], dtype='float32')
    tmp[nz//2:3*nz//2, n//2:3*n//2] = psi
    [x,y] = np.meshgrid(np.fft.rfftfreq(2*n),np.fft.fftfreq(2*nz))
    shift = np.exp(-2*np.pi*1j*(x*p[1]+y*p[0]))
    res0 = np.fft.irfft2(shift*np.fft.rfft2(tmp))
    res = res0[nz//2:3*nz//2, n//2:3*n//2]
    return res

if __name__ == "__main__":
    binning = 1
    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]
    data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(name+'_bin1'+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
    np.save(name+'_abin'+str(binning)+str(0),data[:nth])
    exit()  
    for j in range(1,ndsets):
        for k in range(0,nth):
            p = skimage.feature.register_translation(data[0+k], data[nth*j+k], upsample_factor=1, space='real', return_error=False)
            print(j,k,theta[k],theta[j*nth+k]-2*np.pi*j,p)
            data[nth*j+k] = apply_shift(data[nth*j+k], p)
        np.save(name+'_abin'+str(binning)+str(j),data[nth*j:nth*(j+1)])    