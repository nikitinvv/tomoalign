import tomopy
import dxchange
import numpy as np
import h5py
import sys
import skimage.feature

##################################### Inputs #########################################################################
file_name = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/ZP_16nmZP_73to66_9100eV_1s_4000prj_001.h5'
ndsets = 1
sino_start =1024
sino_end = 2048
theta_start = 0
theta_end = 4000
flat_field_norm = True
flat_field_drift_corr = True  # Correct the intensity drift
remove_rings = True
binning = 2
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
    prj, flat, dark, theta = dxchange.read_aps_32id(
        file_name, sino=(sino_start, sino_end), proj=(theta_start,theta_end*ndsets))
    # preprocess
    prj = preprocess_data(prj, flat, dark, FF_norm=flat_field_norm, remove_rings=remove_rings,
                          FF_drift_corr=flat_field_drift_corr, downsapling=binning)

    #dxchange.write_tiff_stack(prj.swapaxes(1,2),'/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/sinostack/rs')
    #print(np.linalg.norm(prj))
    np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/prjbin2p1',prj)        
    np.save('/data/staff/tomograms/vviknik/tomoalign_vincent_data/ZP/theta',theta)  
        