
import dxchange
import numpy as np
import dxchange
from scipy.ndimage import rotate
import sys
import matplotlib.pyplot as plt
if __name__ == "__main__":

    in_file = sys.argv[1]
    idF = in_file.find('rect')
    out_file = in_file[:idF-1]+'rotated'+in_file[idF-1:]
    print('rotate',out_file)
    #mul=2
    binning=0
    data = dxchange.read_tiff_stack(in_file+'/r_00000.tiff', ind=range(0, 2048//pow(2,binning)))

    print(data.shape)
    #data = data.swapaxes(0,1)
    #data = data[:,:,(128*2)//pow(2,binning):(600*2)//pow(2,binni1)]
    data = rotate(data, 51, reshape=False, axes=(1, 2), order=1)
    #data = data[:,:,300//pow(2,binning):-300//pow(2,binning)]
    data = data.swapaxes(0,2)
    data = rotate(data, 34, reshape=False, axes=(1, 2), order=1)
    data = data.swapaxes(0,2)
    dxchange.write_tiff_stack(data, out_file+'/r', overwrite=True)
    
    #bin2
    plt.imsave(out_file+'z1.png',data[1112//pow(2,binning)],vmin=-0.003,vmax=0.006,cmap='gray')
    plt.imsave(out_file+'z2.png',data[1076//pow(2,binning)],vmin=-0.003,vmax=0.006,cmap='gray')
    plt.imsave(out_file+'z3.png',data[1092//pow(2,binning)],vmin=-0.003,vmax=0.006,cmap='gray')

    #bin2    
    plt.imsave(out_file+'z1.png',data[1112//pow(2,binning)],vmin=-0.002,vmax=0.004,cmap='gray')
    plt.imsave(out_file+'z2.png',data[1076//pow(2,binning)],vmin=-0.002,vmax=0.004,cmap='gray')
    plt.imsave(out_file+'z3.png',data[1092//pow(2,binning)],vmin=-0.002,vmax=0.004,cmap='gray')