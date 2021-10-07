import dxchange
import numpy as np
import tomoalign
import sys

nthetas={
'482': 2160,
'483': 2160,
'484': 2160,
'485': 2160,
'486': 2160,
'487': 2160,
'491': 1262,
'492': 1192,
'157': 2880,
'158': 2160,
'159': 2160,
'160': 2160,
'161': 2160,
}


nthetas={
'482': 2160,
'483': 2160,
'484': 2160,
'485': 2160,
'486': 2160,
'487': 2160,
'491': 1440,
'492': 1440,
'157': 499,
'158': 512,
'159': 506,
'160': 518,
'161': 499,
}

######################################################
file_name = sys.argv[1]
center = centers[file_name[-6:-3]]
ntheta = nthetas[file_name[-6:-3]]

ngpus = 8
pnz = 16 # chunk size for slices
ptheta = 20 # chunk size for angles

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
