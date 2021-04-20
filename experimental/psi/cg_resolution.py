import dxchange
import numpy as np
import sys
import tomoalign

ngpus = 4

if __name__ == "__main__":
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
    nitercg = 64
    pnz = 80
    center = 272

    data = np.ascontiguousarray(data[2::2])
    theta = np.ascontiguousarray(theta[2::2])
   
    res = tomoalign.cg(data, theta, pprot, pnz, center, ngpus, nitercg, padding=False)
    dxchange.write_tiff_stack(res['u'], fname+'/results_cgp2/u/r', overwrite=True)
    
            