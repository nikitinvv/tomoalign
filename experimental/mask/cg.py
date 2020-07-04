import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from timing import tic,toc
import gc
import scipy.ndimage as ndimage
matplotlib.use('Agg')
matplotlib.use('Agg')
centers={
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run1_8keV_phase_interlaced_100prj_per_rot_1201prj_1s_006': 1204,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run21_40min_8keV_phase_interlaced_1201prj_1s_012': 1187,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_PBI_new_ROI_8keV_phase_interlaced_2000prj_1s_042':  1250,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_ROI2_8keV_phase_interlaced_1201prj_0.5s_037':  1227,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PAN_PI_PBI_new_ROI_8keV_phase_interlaced_1201prj_0.5s_041': 1248,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024': 1202,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_PAN_8keV_phase_721prj_0.5s_045': 1244,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_PAN_8keV_phase_interlaced_1201prj_1s_046': 1241,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_3h_8keV_phase_interlaced_1201prj_0.5s_047_049': 1226,
'/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/PVDF_3h_ROI2_8keV_phase_interlaced_1201prj_0.5s_047_050': 1209,
}

ngpus = 4

if __name__ == "__main__":

    ndsets = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]   
    
    niter = 128
    binning=1
    data = np.zeros([ndsets*nth,2048//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(name+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(name+'_theta'+str(k)+'.npy').astype('float32')
        [ntheta, nz, n] = data.shape  # object size n x,y
    [ntheta, nz, n] = data.shape  # object size n x,y

    data[np.isnan(data)]=0            
    data-=np.mean(data)
    mmin,mmax = find_min_max(data)
    # pad data    
    ne = 3456//pow(2,binning)    
    #ne=n
    center = centers[sys.argv[3]]+(ne//2-n//2)*pow(2,binning)        
    pnz = 8*pow(2,binning)  # number of slice partitions for simultaneous processing in tomography
    ptheta = 60

       
    u = np.zeros([nz, ne, ne], dtype='float32')
    psi = data.copy()
    lamd = np.zeros([ntheta, nz, n], dtype='float32')
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
       
    with tc.SolverTomo(theta, ntheta, nz, ne, pnz, center/pow(2, binning), ngpus) as tslv:
        u = tslv.cg_tomo_batch(pad(psi,ne,n), u, niter)     
        dxchange.write_tiff_stack(
                        u[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],  name+'/cg_'+'_'+str(ntheta)+'/rect''/r', overwrite=True)
        