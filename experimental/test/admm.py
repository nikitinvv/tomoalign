import dxchange
import numpy as np
import tomoalign
import sys


######################################################
file_name = sys.argv[1]
center = ???
ntheta = 900

ngpus = 1
pnz = 16 # chunk size for slices
ptheta = 20 # chunk size for angles

# read data
prj = dxchange.read_tiff_stack(f'{file_name[:-3]}/data/d_00000.tiff', ind = range(ntheta))
theta = np.load(file_name[:-3]+'/data/theta.npy')  
nz, n = prj.shape[1:]

niteradmm = [75,36,18]  # number of iterations in the ADMM scheme
startwin = [200,100,50] # starting window size in optical flow estimation
stepwin = [2,2,2] # step for decreasing the window size in optical flow estimtion
    
res = tomoalign.admm_of_levels(
    prj, theta, pnz, ptheta, center, ngpus, niteradmm, startwin, stepwin, file_name[:-3]+'/tmp/', padding=True)

dxchange.write_tiff_stack(
    res['u'], file_name[:-3]+'/results_admm/u/r', overwrite=True)
dxchange.write_tiff_stack(
    res['psi'], file_name[:-3]+'/results_admm/psi/r', overwrite=True)
