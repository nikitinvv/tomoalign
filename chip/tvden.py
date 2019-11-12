import dxchange
import numpy as np
import sys
import skimage

if __name__ == "__main__":

    in_file = sys.argv[1]
    idF = in_file.find('rotated')
    out_file = in_file[:idF-1]+'denoised'+in_file[idF-1:]
    print(in_file)
    data = dxchange.read_tiff_stack(in_file+'/rect212_00000.tiff',ind=range(0,256))
    datanew = skimage.restoration.denoise_tv_chambolle(data, weight=0.001, eps=0.00001, n_iter_max=100, multichannel=False)

    dxchange.write_tiff_stack(datanew,out_file,overwrite=True)
