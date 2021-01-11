
import dxchange
import numpy as np
import dxchange
from scipy.ndimage import rotate
from scipy.ndimage import median_filter
import sys
import matplotlib.pyplot as plt
import concurrent.futures as cf
import threading
from functools import partial

def rotate_thread(data,ang,order,ids):
    data[ids] = rotate(data[ids], ang, reshape=False, axes=(0, 1), order=order)
    
def rotate_batch(data,ang,order):
    print('rotate',ang)
    with cf.ThreadPoolExecutor() as e:
        # update flow in place
        e.map(partial(rotate_thread, data,
                       ang,order), range(0, data.shape[0]))       
    return data

if __name__ == "__main__":

    in_file = sys.argv[1]
    order = int(sys.argv[2])
    iter = int(sys.argv[3])
    binning = int(sys.argv[4])
    #idF = in_file.find('rect')
    out_file = in_file+'rotated'+str(order)+'_'+str(iter)+'_'+str(binning)
    print('rotate',out_file)
    #data = dxchange.read_tiff_stack(in_file+'/data/of_recon/recon/iter'+str(iter)+'_00000.tiff', ind=range(0, 2048//pow(2,binning)))
    data = dxchange.read_tiff_stack(in_file+'/results_admm/u/r_00000.tiff', ind=range(0, 2048//pow(2,binning)))

    data = rotate_batch(data, 51, order=order)
    data = data.swapaxes(0,2)
    data = rotate_batch(data, 34, order=order)
    data = data.swapaxes(0,2)

    data = data[450//pow(2,binning):962//pow(2,binning)]
    # data = data[200//pow(2,binning):1224//pow(2,binning)]
    dxchange.write_tiff_stack(data, out_file+'/r', overwrite=True)
    
    data=dxchange.read_tiff_stack(out_file+'/r_00000.tiff', ind=[(691-450)//pow(2,binning),(658-450)//pow(2,binning),(672-450)//pow(2,binning)])

    # #bin2
    # plt.imsave(out_file+'z1.png',data[1112//pow(2,binning)],vmin=-0.003,vmax=0.006,cmap='gray')
    # plt.imsave(out_file+'z2.png',data[1076//pow(2,binning)],vmin=-0.003,vmax=0.006,cmap='gray')
    # plt.imsave(out_file+'z3.png',data[1092//pow(2,binning)],vmin=-0.003,vmax=0.006,cmap='gray')

    #bin0 cutes    
    plt.imsave(out_file+'z1.png',data[0],vmin=-0.001,vmax=0.002,cmap='gray')
    plt.imsave(out_file+'z2.png',data[1],vmin=-0.001,vmax=0.002,cmap='gray')
    plt.imsave(out_file+'z3.png',data[2],vmin=-0.001,vmax=0.002,cmap='gray')