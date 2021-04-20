import dxchange
import numpy as np
import sys
import tomoalign

ngpus = 4

if __name__ == "__main__":
    alpha=np.float32(sys.argv[1])
    fname = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/d'
    ndsets = 1
    nth = 380
    
    binning = 1
    data = np.zeros([ndsets*nth,320,544],dtype='float32')
    theta = np.zeros(ndsets*nth,dtype='float32')
    for k in range(ndsets):
        data[k*nth:(k+1)*nth] = np.load(fname+'_bin'+str(binning)+str(k)+'.npy').astype('float32')                                   
        theta[k*nth:(k+1)*nth] = np.load(fname+'_theta'+str(k)+'.npy').astype('float32')
    data[np.isnan(data)]=0           
    #data-=np.mean(data)


    ngpus = 4
    pprot = 190
    nitercg = 32
    pnz = 80
    center = 272
    ptheta = 10
    niteradmm = [128, 56]  # number of iterations in the ADMM scheme
    startwin = [160, 64]  # starting window size in optical flow estimation
    # step for decreasing the window size in optical flow estimtion
    stepwin = [1, 1, 1]
    center = 272
    
  
    res = tomoalign.admm_of_reg_levels(data, theta, pnz, ptheta, center, alpha, ngpus, niteradmm, startwin, stepwin, fname, padding=False)   
    dxchange.write_tiff_stack(res['u'], fname+'/results_admm_reg'+str(alpha)+'/u/r', overwrite=True)
    np.save(fname+'/results_admm_reg'+str(alpha)+'/flow.npy',res['flow'])
        
            