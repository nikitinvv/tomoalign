import dxchange
import numpy as np
import sys
import tomoalign
import scipy.ndimage as ndimage
centers = {
    '/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090': 1224,#1277,
}

if __name__ == "__main__":

    ndsets = 6
    nth = 120
    fname = '/local/data/vnikitin/ZP/Kenan_ZP_9100eV_interlaced_-14to16deg_3s_090'
    
    binning = 1
    data = np.zeros([ndsets*nth, 2048//pow(2, binning),
                     2448//pow(2, binning)], dtype='float32')
    theta = np.zeros(ndsets*nth, dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin' +
                                        str(binning)+str(k)+'.npy').astype('float32')
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta' +
                                         str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0    
    data-=np.mean(data)
    ngpus = 8
    pnz = 4
    ptheta = 10
    niteradmm = [48]  # number of iterations in the ADMM scheme
    # starting window size in optical flow estimation
    startwin = [512]
    # step for decreasing the window size in optical flow estimtion
    stepwin = [8,8,8]
    center = (centers[fname])/pow(2, binning)




    fname += '/densenew1224'+'_'+str(binning)+'p2'
    
    data3 = np.ascontiguousarray(data[1::2].astype('float32'))
    theta3 = np.ascontiguousarray(theta[1::2].astype('float32'))
    
    res = tomoalign.admm_of_levels(
        data3, theta3, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)
    
    dxchange.write_tiff_stack(
        res['u'].swapaxes(0,1), fname+'/results_admm/u/r', overwrite=True)
    dxchange.write_tiff_stack(
        res['psi'], fname+'/results_admm/psi/r', overwrite=True)
    np.save(fname+'/results_admm/flow.npy', res['flow'])
