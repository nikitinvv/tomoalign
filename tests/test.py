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
    ngpus = 4  # number of gpus
    nitercg = 64  # number of iterations in the cg scheme
    niteradmm = 64  # number of iterations in the ADMM scheme
    startwin = n # starting window size in optical flow estimation
    stepwin = 4 # step for decreasing the window size in optical flow estimtion
    res = tomoalign.pcg(data, theta, pprot, pnz, center, ngpus, nitercg)
    dxchange.write_tiff(res['u'], 'data/cg_recon/recon/iter' +
                        str(nitercg), overwrite=True)

    res = tomoalign.admm_of(data, theta, pnz, ptheta,
                            center, ngpus, niteradmm, startwin, stepwin)
    dxchange.write_tiff(res['u'], 'data/of_recon/recon/iter' +
                        str(niteradmm), overwrite=True)
