
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
    binning = np.int(sys.argv[2])
    
    data = dxchange.read_tiff_stack(in_file+'/r_00000.tiff', ind=range(0, 1536//pow(2,binning)))
    data = rotate(data, -42, reshape=False, axes=(1, 2), order=1)
    data = data.swapaxes(0,1)
    data = rotate(data, 37.2, reshape=False, axes=(1, 2), order=1)
    data = data.swapaxes(0,1)
    #data = data[512//pow(2,binning):-512//pow(2,binning)]
    dxchange.write_tiff_stack(data, out_file+'/r', overwrite=True)
    out_file2 = in_file[:idF-1]+in_file[idF:]
    dxchange.write_tiff(data[960//pow(2,binning)], in_file[:idF-1]+'results/'+in_file[idF-1:],overwrite=True)
