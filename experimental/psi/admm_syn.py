import dxchange
import numpy as np
import sys
import tomoalign
import struct
ngpus = 4

if __name__ == "__main__":
    fname = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/data_syn/tomo_delta__ram-lak_freqscl_1.00'
    thetaname = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/theta.bin'
    ndsets = 1
    nth = 400
    
    data = dxchange.read_tiff_stack(fname+'_0001.tif',ind = np.arange(1,ndsets*nth+1))/1e4#[:]/32768.0/2.0
    fid = open(thetaname, 'rb')
    theta = np.float32(np.reshape(struct.unpack(
            ndsets*nth*'f', fid.read(nth*4)), [nth]))
    theta = np.ascontiguousarray((theta/180*np.pi).astype('float32'))
    data = np.ascontiguousarray(data.astype('float32'))
    
    print(theta)

    print(np.linalg.norm(data[-1]))
    print(data.shape)
    print(theta.shape)
    ngpus = 4
    pprot = 100
    nitercg = 64
    pnz = 16
    center = 144
    ptheta = 10
    niteradmm = [60]  # number of iterations in the ADMM scheme
    startwin = [128]  # starting window size in optical flow estimation
    # step for decreasing the window size in optical flow estimtion
    stepwin = [2]
    print(data.shape)
    theta = np.ascontiguousarray(theta)
    res = tomoalign.cg(data, theta, pprot, pnz, center, ngpus, nitercg, padding=False)
    dxchange.write_tiff_stack(res['u'], fname+'/results_cg2/u/r', overwrite=True)
   
    res = tomoalign.admm_of_reg_levels(data, theta, pnz, ptheta, center,2e-5, ngpus, niteradmm, startwin, stepwin, fname, padding=False)   
    dxchange.write_tiff_stack(res['u'], fname+'/results_admm_reg2/u/r', overwrite=True)
    
            