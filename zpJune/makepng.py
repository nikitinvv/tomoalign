import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from timing import tic,toc
import gc
import scipy.ndimage as ndimage
import cv2

name = sys.argv[1]
nameout = sys.argv[2]
u = dxchange.read_tiff_stack(name,ind=np.arange(0,1024))
plt.imsave(nameout+'z.png',u[u.shape[0]//2],vmin=float(sys.argv[3]),vmax=float(sys.argv[4]),cmap='gray')
plt.imsave(nameout+'y.png',u[:,u.shape[1]//2],vmin=float(sys.argv[3]),vmax=float(sys.argv[4]),cmap='gray')
plt.imsave(nameout+'x.png',u[:,:,u.shape[2]//2],vmin=float(sys.argv[3]),vmax=float(sys.argv[4]),cmap='gray')
