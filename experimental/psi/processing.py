import dxchange
import numpy as np
import h5py
import sys
import skimage.feature
import struct


name = '/data/staff/tomograms/vviknik/tomoalign_vincent_data/psi/'

######################################################################################################################
Nproj  = 380
N = 544
Ns = 320
binning  = 1
if __name__ == "__main__":
    for k in range(1):
        # preprocess
        fid = open(name+'sino2.bin', 'rb')
        tomo = np.float32(np.reshape(struct.unpack(
            Nproj*N*Ns*'f', fid.read(Nproj*N*Ns*4)), [Nproj, N, Ns])).swapaxes(1, 2)
        fid = open(name+'theta.bin', 'rb')
        theta = np.float32(np.reshape(struct.unpack(
            Nproj*'f', fid.read(Nproj*4)), [Nproj]))
        fid = open(name+'rec_0.bin', 'rb')
        rec = np.float32(np.reshape(struct.unpack(
            N*N*Ns*'f', fid.read(N*N*Ns*4)), [Ns, N, N]))

        theta = theta/180*np.pi
        print(tomo.shape)
        np.save(name+'d_bin'+str(binning)+str(k),tomo)        
        np.save(name+'d_theta'+str(k),theta)  
        np.save(name+'d_rec'+str(k),rec)  
        dxchange.write_tiff_stack(tomo,name+'data/d')
        dxchange.write_tiff_stack(rec,name+'recpsi/r')