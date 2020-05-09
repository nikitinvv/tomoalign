
import dxchange
import numpy as np
import dxchange
from scipy.ndimage import rotate
import sys

if __name__ == "__main__":

    in_file = sys.argv[1]
    idF = in_file.find('rect')
    out_file = in_file[:idF-1]+'rotated'+in_file[idF-1:]
    print('rotate',out_file)
    #mul=2
    binning=1
    data = dxchange.read_tiff_stack(in_file+'/r_00000.tiff', ind=range(256, 1024-128))

    print(data.shape)
    #data = data.swapaxes(0,1)
    #data = data[:,:,(128*2)//pow(2,binning):(600*2)//pow(2,binning)]
    data = rotate(data, -29.5, reshape=False, axes=(1, 2), order=1)
    #data = data[:,:,300//pow(2,binning):-300//pow(2,binning)]
    #data = data.swapaxes(1,2)
    #data = rotate(data, -0.4, reshape=False, axes=(1, 2), order=3)
    data = data.swapaxes(0,2)
    data = rotate(data, -52, reshape=False, axes=(1, 2), order=1)
    data = data.swapaxes(0,1)
    dxchange.write_tiff_stack(data[512+100:-512], out_file+'/r', overwrite=True)
    #out_file2 = in_file[:idF-1]+in_file[idF:]
    print(data[data.shape[0]//2].shape)
    #print(in_file[:idF-1]+'results/'+in_file[idF-1:-1])

    dxchange.write_tiff(data[512+100+61], in_file[:idF-1]+'results/'+in_file[idF-1:-1],overwrite=True)
