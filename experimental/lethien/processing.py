import tomopy
import dxchange
import numpy as np
import h5py
import sys


nthetas={
'482': 2160,
'483': 2160,
'484': 2160,
'485': 2160,
'486': 2160,
'487': 2160,
'491': 1440,
'492': 1440,
'157': 2880,
'158': 2160,
'159': 2160,
'160': 2160,
'161': 2160,
'162': 2160,
'163': 2160,
'164': 2160,
'165': 2160,
'166': 2160,
'167': 2160,
}


######################################################
file_name = sys.argv[1]
ntheta = nthetas[file_name[-6:-3]]


sino_start = 512
sino_end = 1536+512


ptheta = 720 # chunk size for reading
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
    dxchange.write_tiff_stack(prj[:,:,246:-246], f'{file_name[:-3]}/data/d', start=ptheta*k, overwrite=True)  

# save theta
np.save(file_name[:-3]+'/data/theta',theta)  
print(theta)
            
