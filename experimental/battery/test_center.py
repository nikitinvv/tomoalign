import numpy as np
import dxchange
import tomoalign

file_name = '/data/staff/tomograms/vviknik/experiments/APS/2021-03/Lethien/Sample1_16nmZP_8keV_2200prj_219.h5'
ntheta = 2200
ngpus = 1
niter = 32
nz = 2 # min two slices 
pnz = 1 # chunk size in z

# read data
prj = dxchange.read_tiff_stack(f'{file_name[:-3]}/data/d_00000.tiff', ind = range(ntheta))
prj = prj[:,prj.shape[1]//2:prj.shape[1]//2+2]
theta = np.load(file_name[:-3]+'/data/theta.npy')  
n = prj.shape[2]

for center in range(n//2-20,n//2+20):
    print(f'check center {center}')
    init = np.zeros([nz,n,n],dtype='float32')
    with tomoalign.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
        u = tslv.cg_tomo_batch(prj, init, niter)
        dxchange.write_tiff(u[0], f'{file_name[:-3]}/try_center/r_{center:03.1f}',overwrite=True)