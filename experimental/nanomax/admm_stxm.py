import dxchange
import numpy as np
import sys
import tomoalign
import dxchange
data_prefix = '/data/staff/tomograms/vviknik/nanomax/'

if __name__ == "__main__":    

    # read object
    n = 384  # object size n x,y
    nz = 320  # object size in z
    ntheta = 166  # number of angles
    pnz = 8  # partial size for nz
    
    theta = np.zeros(ntheta, dtype='float32')

    data = np.zeros([ntheta,nz,n],dtype='float32')
    for k in range(ntheta):
        theta[k] = np.load(data_prefix+'datanpy/theta128sorted_'+str(k)+'.npy')
        data[k] = dxchange.read_tiff(f'{data_prefix}/reccrop_align_azat/psiangle/r{k}.tiff')   
    ngpus = 4
    pnz = 8
    ptheta = 2
    niteradmm = [128]  # number of iterations in the ADMM scheme
    # starting window size in optical flow estimation
    startwin = [256]
    # step for decreasing the window size in optical flow estimtion
    stepwin = [2]
    center = 192
    # data = dxchange.read_tiff_stack(data_prefix+'/reccrop_align_azat/psiangle/r_00000.tiff',ind=range(ntheta))
    
    data_prefix+= 'rectomoalign_azat/'

    res = tomoalign.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, data_prefix,padding=False)

    dxchange.write_tiff_stack(
        res['u'], data_prefix+'/results_admm/u/r', overwrite=True)
    dxchange.write_tiff_stack(
        res['psi'], data_prefix+'/results_admm/psi/r', overwrite=True)
    np.save(data_prefix+'/results_admm/flow.npy', res['flow'])
