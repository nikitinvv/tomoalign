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
    # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
    stepwin = 4
    ngpus = 4  # number of gpus
    nitercg = 64  # number of iterations in the cg scheme
    niteradmm = 64  # number of iterations in the ADMM scheme
    niterinner = 4  # number of inner ADMM iterations

    ucg = tomoalign.pcg(data, theta, pprot, pnz, center, ngpus, nitercg)
    dxchange.write_tiff(ucg, 'data/cg_recon/recon/iter' +
                        str(nitercg), overwrite=True)

    uof = tomoalign.admm_of(data, theta, pnz, ptheta,
                            center, stepwin, ngpus, niteradmm, niterinner)
    dxchange.write_tiff(uof, 'data/of_recon/recon/iter' +
                        str(niteradmm), overwrite=True)
