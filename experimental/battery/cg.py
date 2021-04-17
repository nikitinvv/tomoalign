import numpy as np
import dxchange
import tomoalign

file_name = '/data/staff/tomograms/vviknik/experiments/APS/2021-03/Lethien/Sample1_16nmZP_8keV_2200prj_219.h5'
ntheta = 2200
ngpus = 4
niter = 32
pnz = 16 # chunk size in z
center = 629

# read data
prj = dxchange.read_tiff_stack(f'{file_name[:-3]}/data/d_00000.tiff', ind = range(ntheta))
theta = np.load(file_name[:-3]+'/data/theta.npy')  
nz,n = prj.shape[1:]

init = np.zeros([nz,n,n],dtype='float32')
with tomoalign.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
    u = tslv.cg_tomo_batch(prj, init, niter, dbg=True)
    dxchange.write_tiff_stack(u, f'{file_name[:-3]}/cg/r',overwrite=True)