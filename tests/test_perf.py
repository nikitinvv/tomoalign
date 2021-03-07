import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign
import scipy.ndimage as ndimage
if __name__ == "__main__":

    n = 512
    ntheta = 1000
    pprot = 100
    # adef = 10
    # tomoalign.gen_cyl_data(n, ntheta, pprot, adef)
    data = dxchange.read_tiff('data/deformed_data.tiff')
    data = ndimage.zoom(data,[ntheta//data.shape[0],n//data.shape[1],n//data.shape[2]])
    [ntheta, nz, n] = data.shape
    
    theta = np.linspace(0, 4*np.pi, ntheta).astype('float32')    
    center = n/2
    pnz = 32  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    ptheta = 100  # number of projections for simultaneus processing by one GPU in the alignment sub-problem
    # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
    ngpus = 1  # number of gpus
    niteradmm = [8]  # number of iterations in the ADMM scheme
    startwin = [n]
    stepwin = [2]
    
    fname = 'data/tmp'
    uof = tomoalign.admm_of_levels(
        data, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, fname)

    dxchange.write_tiff(uof['u'], 'data/of_recon/recon/iter' +
                        str(niteradmm), overwrite=True)
