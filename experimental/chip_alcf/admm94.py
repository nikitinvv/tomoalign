import dxchange
import numpy as np
import sys
import tomoalign
import scipy.ndimage as ndimage


if __name__ == "__main__":

    nsets = np.int(sys.argv[1])
    ndsets = np.int(sys.argv[2])
    nth = np.int(sys.argv[3])
    start = np.int(sys.argv[4])
    name = sys.argv[5]

    binning = 0
    data = np.zeros([nsets*ndsets*nth, (2048)//pow(2, binning),
                     (2448)//pow(2, binning)], dtype='float32')
    theta = np.zeros(nsets*ndsets*nth, dtype='float32')
    strs = ['094']
    for j in range(nsets):                
        name0 = name[:-3]+strs[j]
        print(name0)
        for k in range(ndsets):
            print(j,k)
            idstart = j*ndsets*nth+k*nth
            data[idstart:idstart+nth] = np.load(name0+'fw_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
            theta[idstart:idstart+nth] = np.load(name0+'_theta'+str(k)+'.npy').astype('float32')
    data[np.isnan(data)] = 0
    data = data[:3900,:-256,:]
    ngpus = 8
    pnz = 8
    ptheta = 10
    niteradmm = [80, 72, 32, 26]  # number of iterations in the ADMM scheme
    niteradmm = [2, 2, 2, 2]  # number of iterations in the ADMM scheme
    # starting window size in optical flow estimation
    startwin = [256, 192, 96, 64]
    # step for decreasing the window size in optical flow estimtion
    stepwin = [1, 1, 1, 2]
    center = 1158.5
    name += '/fw'+str(len(theta))+'_'+str(start)+'/'
    fname_fast = ?
    res = tomoalign.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, name,fname_fast,padding=True)

    dxchange.write_tiff_stack(
        res['u'], name+'/results_admm/u/r', overwrite=True)
