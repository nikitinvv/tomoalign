from scipy import ndimage
import numpy as np
import time

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
       return time.time() - startTime_for_tictoc

def find_min_max(data):
    """Find min and max values according to histogram"""
    
    mmin = np.zeros(data.shape[0],dtype='float32')
    mmax = np.zeros(data.shape[0],dtype='float32')
    
    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:],1000)
        stend = np.where(h>np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]        
        mmin[k] = e[st]
        mmax[k] = e[end+1]
        # mmin[k] = np.min(data[k])
        # mmax[k] = np.max(data[k])
    #print(mmin,mmax)
    return mmin,mmax

def paddata(data,ne,n):
    """Pad tomography projections"""
    
    datae = np.zeros([data.shape[0],data.shape[1],ne],dtype='float32')
    datae[:,:,ne//2-n//2:ne//2+n//2]=data
    datae[:,:,:ne//2-n//2]=datae[:,:,ne//2-n//2:ne//2-n//2+1]
    datae[:,:,ne//2+n//2:]=datae[:,:,ne//2+n//2-1:ne//2+n//2]
    return datae

def unpaddata(data,ne,n):
    """Unpad tomography projections"""    
    
    return data[:,:,ne//2-n//2:ne//2+n//2]

def unpadobject(f,ne,n):
    """Unpad 3d object"""    
    
    return f[:,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2]


def interpdense(u,psi,lamd,flow):
    """Interpolate ADMM variables to a twice dense grid"""    
    
    unew = ndimage.zoom(u,2,order=1)
    psinew = ndimage.zoom(psi,(1,2,2),order=1)
    lamdnew = ndimage.zoom(lamd,(1,2,2),order=1)
    flownew = ndiage.zoom(flow,(1,2,2,1),order=1)/2    
    return unew,psinew,lamdnew,flownew






def mload(fname, *args):
    res = []
    for k in range(0,len(args),2):
#        print('load',args[k+1])
        res.append(np.load(fname+'/'+args[k+1]+'.npy'))
    if len(res)==1: 
        res = res[0]    
    else:
         res = tuple(res)            
    return res

def mdump(fname, *args):
    res = []
    for k in range(0,len(args),2):
        #print('save',args[k+1])
        np.save(fname+'/'+args[k+1]+'.npy',args[k])
        #print(args[k].shape)
        res.append(None)        
    if len(res)==1: 
        res = res[0]    
    else:
         res = tuple(res)            
    return res

def munload(fname, *args):
    res = []
    for k in range(0,len(args),2):
        #print('unload',args[k+1])
        res.append(None)
    if len(res)==1: 
        res = res[0]    
    else:
         res = tuple(res)            
    return res
