import tomopy
import dxchange
import numpy as np
import h5py
import sys
import skimage.feature

##################################### Inputs #########################################################################
file_name = '/home/beams/VNIKITIN/tomoalign/tomoalign/3Dstd_new/3Dstd_14nmZP_9700eV_interlaced_2000prj_200proj_per_rot_1s_092.h5'
ndsets = 10
sino_start = 0#1024-512
sino_end = 2048#1024+512
theta_start = 0
theta_end = 200
flat_field_norm = True
flat_field_drift_corr = True  # Correct the intensity drift
remove_rings = True
binning = 1
######################################################################################################################


def preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings, FF_drift_corr=flat_field_drift_corr, downsapling=binning):

    if FF_norm:  # dark-flat field correction
        prj = tomopy.normalize(prj, flat, dark)
    if FF_drift_corr:  # flat field drift correction
        prj = tomopy.normalize_bg(prj, air=100)
    prj[prj <= 0] = 1  # check dark<data
    prj = tomopy.minus_log(prj)  # -logarithm
    if remove_rings:  # remove rings
        prj = tomopy.remove_stripe_fw(
            prj, level=7, wname='sym16', sigma=1, pad=True)
    if downsapling > 0:  # binning
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)
    return prj


if __name__ == "__main__":
    # read data
    prj = np.zeros([200*ndsets,(sino_end-sino_start)//pow(2,binning),2448//pow(2,binning)],dtype='float32')
    theta = np.zeros(200*ndsets,dtype='float32')
    for k in range(ndsets):
        prj0, flat, dark, theta0 = dxchange.read_aps_32id(
            file_name, sino=(sino_start, sino_end), proj=(theta_start,theta_end))        
        prj0 = preprocess_data(prj0, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings,
                          FF_drift_corr=flat_field_drift_corr, downsapling=binning)
        prj[k*200:(k+1)*200] = prj0
        theta[k*200:(k+1)*200] = theta0    
        print(k)
    #prj=prj[:,:,200//pow(2,binning):-200//pow(2,binning)]

    # for j in range(1,ndsets):
    #     for k in range(0,200):
    #         p = skimage.feature.register_translation(prj[0+k], prj[200*j+k], upsample_factor=1, space='real', return_error=False)
    #         print(j,k,p)
    #         prj[200*j+k] = np.real(pt.deformation.fourier_shift(prj[200*j+k], p))
    np.save('prjbin1',prj)        
    np.save('theta',theta)  
        