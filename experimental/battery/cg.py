import numpy as np
import dxchange
import tomoalign
import sys

centers={
'219': 1247,
'220': 1214,
'221': 1216,
'222': 1198,
}


nthetas={
'219': 2200,
'220': 1400,
'221': 3000,
'222': 2200,
}


######################################################
file_name = sys.argv[1]
center = centers[file_name[-6:-3]]
ntheta = nthetas[file_name[-6:-3]]

ngpus = 8
niter = 64
pnz = 16 # chunk size in z

# read data
prj = dxchange.read_tiff_stack(f'{file_name[:-3]}/data/d_00000.tiff', ind = range(ntheta))
theta = np.load(file_name[:-3]+'/data/theta.npy')  
nz,n = prj.shape[1:]

init = np.zeros([nz,n,n],dtype='float32')
with tomoalign.SolverTomo(theta, ntheta, nz, n, pnz, center, ngpus) as tslv:
    u = tslv.cg_tomo_batch(prj, init, niter, dbg=False)
    dxchange.write_tiff_stack(u, f'{file_name[:-3]}/cg/r',overwrite=True)
