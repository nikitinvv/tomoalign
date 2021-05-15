import tomopy
import dxchange
import numpy as np
import h5py
import sys




######################################################
file_name = sys.argv[1]
ntheta = 900

sino_start = 0
sino_end = 800

ptheta = 100 # chunk size for reading
binning = 0

for k in range(int(np.ceil(ntheta/ptheta))):
    print(k)
    prj, flat, dark, theta = dxchange.read_aps_32id(
        file_name, sino=(sino_start, sino_end), proj=(ptheta*k,min(ntheta,ptheta*(k+1))))

    prj = tomopy.normalize(prj, flat, dark)
    prj[prj <= 0] = 1 
    prj = tomopy.minus_log(prj) 
    # prj = tomopy.remove_stripe_fw(
            # prj, level=7, wname='sym16', sigma=1, pad=True)
    # prj = tomopy.remove_stripe_ti(prj,2)
    prj = tomopy.downsample(prj, level=binning)
    prj = tomopy.downsample(prj, level=binning, axis=1)
    
    # save data
    dxchange.write_tiff_stack(prj, f'{file_name[:-3]}/data/d', start=ptheta*k, overwrite=True)  

# save theta
np.save(file_name[:-3]+'/data/theta',theta)  
print(theta)
            
