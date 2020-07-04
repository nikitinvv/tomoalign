import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign

if __name__ == "__main__":

    n = 256
    ntheta = 384
    pprot = 192
    # adef = 10
    # tomoalign.gen_cyl_data(n, ntheta, pprot, adef)
    data = dxchange.read_tiff('data/deformed_data.tiff')
    theta = np.load('data/theta.npy')
    [ntheta, nz, n] = data.shape

    center = n/2
    pnz = n  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    ptheta = 16  # number of projections for simultaneus processing by one GPU in the alignment sub-problem
    ngpus = 1  # number of gpus
    nitercg = 64  # number of iterations in the cg scheme
    niteradmm = [48, 24]  # number of iterations in the ADMM scheme
    startwin = [n//2, n//2]  # starting window size in optical flow estimation
    # step for decreasing the window size in optical flow estimtion
    stepwin = [2, 2]

    # ADMM OF
    res = tomoalign.admm_of_levels(data, theta, pnz, ptheta,
                                       center, ngpus, niteradmm, startwin, stepwin)
    dxchange.write_tiff(res['u'], 'data/of_recon_layers/recon/iter' +
                        str(niteradmm), overwrite=True)
    # # ADMM OFTV
    alpha = 1e-3
    res = tomoalign.admm_of_reg_levels(data, theta, pnz, ptheta,
                                       center, alpha, ngpus, niteradmm, startwin, stepwin)
    dxchange.write_tiff(res['u'], 'data/of_recon_reg_layers/recon/iter' +
                        str(niteradmm), overwrite=True)
