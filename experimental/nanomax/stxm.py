#import necessary libraries
import numpy as np
import h5py as hp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
import glob
import random
#Import reference projection to determine array sizes
f = hp.File('/data/staff/tomograms/vviknik/nanomax/data/scan_000{}.h5'.format(204),'r')
measured = f['measured']
data = np.array(measured['diffraction_patterns'])
mask = np.array(measured['mask'])
theta = np.array(measured['angle_deg'])
positions = np.array(measured['positions_um'])
pxsz = f['reconstructed/pixel_size_nm'].value

#Horizontal and vertical elements in the scanning grid
n = 169
nz = 81
    
#Find the scanning ranges to define the image size
Y = (np.max(positions[:,0])-np.min(positions[:,0]))
X = (np.max(positions[:,1])-np.min(positions[:,1]))

#Define the gridsize for STXM projections based on the horizontal (more densely sampled) size of the scanning grid .
#Consider changing for binned data according to the number of scanning points to reduce the image size.
nx = 169*2 #(do *2 to avoid interpolation artefacts)
ny = np.int(nx*Y/X)

#Define binning for scanning points
bin_x = 1 #horizontal binning
bin_y = 1 #vertical binning

#Define indices of the first scanning position in each row
scan_id = np.linspace(0,n*nz,nz,endpoint=False).astype(int)
#Select every bin_y-th row
scan_id_array = scan_id[0::bin_y]
#Read sorted scanning positions per angular view
data_path = np.sort(glob.glob('/data/staff/tomograms/vviknik/nanomax/data/*'))[1:]
theta_array = np.zeros((data_path.shape[0]))
dpc = np.zeros((data_path.shape[0],ny,nx))

for idx in range(0,data_path.shape[0]):
    #read data of the idx-th scanning position
    print(data_path[idx])
    f = hp.File('{}'.format(data_path[idx]),'r')
    measured = f['measured']
    mask = np.array(measured['mask'])
    data = np.array(measured['diffraction_patterns'])
    positions = np.array(measured['positions_um'])
    theta = np.array(measured['angle_deg'])
    theta_array[idx] = theta
    
    #Bin data in vertical direction
    if bin_y > 1:
        data_bin = np.zeros((scan_id_array.shape[0]*n,data.shape[1],data.shape[2]))
        positions_bin = np.zeros((scan_id_array.shape[0]*n,2))
        jcount = 0
        for j in scan_id_array:
            data_bin[jcount:jcount+n] = data[j:j+n]
            positions_bin[jcount:jcount+n,0] = positions[j:j+n,0]
            positions_bin[jcount:jcount+n,1] = positions[j:j+n,1]
            jcount = jcount+n
        data = data_bin
        positions = positions_bin
        
    #Bin data in horizontal direction
    data = data[0::bin_x,:]*mask
    positions = positions[0::bin_x,:]
    
    #Define beam-stop mask for dark-field contrast
    beam_stop = np.ones(mask.shape)
    beam_stop[50:78,50:78] = 0
    
    #Initialize arrays for three alternative contrasts
    grad_x = np.zeros((data.shape[0]))
    grad_y = np.zeros((data.shape[0]))
    bright = np.zeros((data.shape[0]))
    scattr = np.zeros((data.shape[0]))
    for i in range(positions.shape[0]):
        a = np.sum(data[i,0:data.shape[-2]//2,:])
        b = np.sum(data[i,data.shape[-1]//2::,:])
        c = np.sum(data[i,:,0:data.shape[-2]//2])
        d = np.sum(data[i,:,data.shape[-1]//2::])
        bright[i] = np.sum(data[i]*(1-beam_stop))
        scattr[i] = np.sum(data[i]*beam_stop)
        grad_x[i] = (a-b)/(a+b+1e-15)
        grad_y[i] = (c-d)/(c+d+1e-15)
    y = positions[:,0]
    x = positions[:,1]
    dy, dx = np.mgrid[np.min(y):np.max(y):ny*1j,np.min(x):np.max(x):nx*1j]
    
    plt.clf()
    plt.plot(dx[::8,::8],dy[::8,::8],'b.')
    idss = random.sample(range(len(x)),len(x)//8)
    plt.plot(x[idss],y[idss],'r.') 
    plt.title(f'{np.min(y)=:.5f} {np.max(y)=:.5f} \n{np.min(x)=:.5f} {np.max(y)=:.5f}')   
    plt.title(f'{np.min(y*1e3/18)=:.2f} {np.max(y*1e3/18)=:.2f} \n{np.min(x*1e3/18)=:.2f} {np.max(y*1e3/18)=:.2f}')   
    plt.savefig(f'/data/staff/tomograms/vviknik/nanomax/pngdp/{idx:03d}.png')    
    points = np.asarray([y,x]).T
    #DPC_x
    dpc_x = griddata(points, grad_x, (dy, dx), method='nearest',fill_value=np.mean(grad_x))
    #DPC_y
    dpc_y = griddata(points, grad_y, (dy, dx), method='nearest',fill_value=np.mean(grad_y))
    #PGM
    dpc[idx] = np.sqrt((dpc_x-np.mean(dpc_x))**2+(dpc_y-np.mean(dpc_y))**2)
    #Absorption
    atten = griddata(points, bright, (dy, dx), method='nearest',fill_value=np.mean(bright))
    #Dark-field
    dark = griddata(points, scattr, (dy, dx), method='nearest',fill_value=np.mean(scattr))
