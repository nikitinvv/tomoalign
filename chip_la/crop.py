import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import glob

if __name__ == "__main__":

    theta = np.arange(-85,85+0.1,0.4).astype('float32')
    print(theta.shape)    
    data = np.zeros([485-59,640,1536],dtype='float32')
    for k in range(0,485-59):        
        print(k+59)
        if (k==(405-59) or k==(415-59) ):
            data[k]=data[k-1].copy()
        else:
            name = "data/all/*fly%03d*" % (k+59)
            data[k] = dxchange.read_tiff(glob.glob(name)[0])[700:700+640,250:250+1536].copy()        
    dxchange.write_tiff_stack(data,'datacropped/data',overwrite=True)        
    