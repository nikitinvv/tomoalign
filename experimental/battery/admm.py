import dxchange
import numpy as np
import tomoalign

file_name = '/data/staff/tomograms/vviknik/experiments/APS/2021-03/Lethien/Sample1_16nmZP_8keV_2200prj_219.h5'

ntheta = 2200
ngpus = 4
center = 629

pnz = 16
ptheta = 10

# read data
prj = dxchange.read_tiff_stack(f'{file_name[:-3]}/data/d_00000.tiff', ind = range(ntheta))
theta = np.load(file_name[:-3]+'/data/theta.npy')  
nz, n = prj.shape[1:]

niteradmm = [96,48,24]  # number of iterations in the ADMM scheme
# niteradmm = [2,2,2]  # number of iterations in the ADMM scheme
startwin = [256,128,64] # starting window size in optical flow estimation
stepwin = [2,2,2] # step for decreasing the window size in optical flow estimtion
    
res = tomoalign.admm_of_levels(
    prj, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, file_name[:-3]+'/tmp/', padding=True)

dxchange.write_tiff_stack(
    res['u'], file_name[:-3]+'/results_admm/u/r', overwrite=True)
dxchange.write_tiff_stack(
    res['psi'], file_name[:-3]+'/results_admm/psi/r', overwrite=True)
