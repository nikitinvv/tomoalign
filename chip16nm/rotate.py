
import dxchange
import numpy as np
import dxchange
from scipy.ndimage import rotate
import sys

if __name__ == "__main__":

    in_file = sys.argv[1]
    outt_file = sys.argv[2]
    idF = in_file.find('rect')
    out_file = in_file[:idF-1]+'rotated'+in_file[idF-1:]
    print('rotate',out_file)
    mul=2
    data = dxchange.read_tiff_stack(in_file+'/r_00000.tiff', ind=range(0, 256*mul))
    data = data.swapaxes(0,1)
    data = rotate(data, 24.5, reshape=False, axes=(1, 2), order=3)
    data = data.swapaxes(0,2)
    data = rotate(data, -32, reshape=False, axes=(1, 2), order=3)
    data = data.swapaxes(0,1)
    dxchange.write_tiff_stack(data, out_file+'/r_0000.tiff', overwrite=True)
    out_file2 = in_file[idF:]
    dxchange.write_tiff_stack(data[315:319], '/data/staff/tomograms/viknik/tomoalign_vincent_data/results/'+outt_file)
