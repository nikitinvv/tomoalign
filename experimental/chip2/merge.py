import dxchange
import numpy as np
import tomocg as tc
import deformcg as dc
import scipy as sp
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from timing import tic, toc
import gc
import scipy.ndimage as ndimage
matplotlib.use('Agg')
centers = {
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073': 1249-456,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_074': 1249-456,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_075': 1249-456,
    '/data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_076': 1249-456,
}
ngpus = 4


def myplot(psi, flow, binning):
    [ntheta, nz, n] = psi.shape

    plt.figure(figsize=(20, 14))
    plt.subplot(3, 4, 1)
    plt.imshow(psi[ntheta//4].real, cmap='gray')

    plt.subplot(3, 4, 2)
    plt.imshow(psi[ntheta//2].real, cmap='gray')
    plt.subplot(3, 4, 3)
    plt.imshow(psi[3*ntheta//4].real, cmap='gray')

    plt.subplot(3, 4, 4)
    plt.imshow(psi[-1].real, cmap='gray')

    plt.subplot(3, 4, 5)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//4]), cmap='gray')

    plt.subplot(3, 4, 6)
    plt.imshow(dc.flowvis.flow_to_color(flow[ntheta//2]), cmap='gray')

    plt.subplot(3, 4, 7)
    plt.imshow(dc.flowvis.flow_to_color(flow[3*ntheta//4]), cmap='gray')
    plt.subplot(3, 4, 8)
    plt.imshow(dc.flowvis.flow_to_color(flow[-1]), cmap='gray')

    if not os.path.exists(name+'/fflowfw_'+'_'+str(ntheta)):
        os.makedirs(
            name+'/fflowfw_'+'_'+str(ntheta))
    plt.savefig(
        name+'/fflowfw_'+'_'+str(ntheta)+'/'+str(kk))
    plt.close()

def find_min_max(data):
    # s = np.std(data,axis=(1,2))
    # m = np.mean(data,axis=(1,2))
    # mmin = m-2*s
    # mmax = m+2*s
    mmin = np.zeros(data.shape[0], dtype='float32')
    mmax = np.zeros(data.shape[0], dtype='float32')

    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:], 1000)
        stend = np.where(h > np.max(h)*0.005)
        st = stend[0][0]
        end = stend[0][-1]
        mmin[k] = e[st]
        mmax[k] = e[end+1]
    return mmin, mmax



if __name__ == "__main__":

    kk = np.int(sys.argv[1])
    nth = np.int(sys.argv[2])
    name = sys.argv[3]

    binnings = [2, 1, 0]
    # ADMM solver
    for il in range(0,2):
        binning = binnings[il]
        data = np.load(name+'_bin' + str(binning)+str(kk)+'.npy').astype('float32')
        [ntheta, nz, n] = data.shape  # object size n x,y
        data[np.isnan(data)] = 0
        mmin, mmax = find_min_max(data)
        res = data.copy()
        with dc.SolverDeform(ntheta, nz, n, 40) as dslv:            
            for idset in range(1, 4):
                psi = np.load(name[:-1]+str(int(name[-1]) +idset)+'_bin'+str(binning)+str(kk)+'.npy').astype('float32')

                # optical flow parameters
                pars = [0.5, 0, 128, 32, 5, 1.1, 4]
                flow = dslv.registration_flow_batch(
                    psi, data, mmin, mmax, None, pars, 40)
                Tpsi = dslv.apply_flow_gpu_batch(psi, flow)
                print(np.linalg.norm(psi-data),np.linalg.norm(Tpsi-data))
                myplot(psi, flow, binning)
                #np.save( name[:-1]+str(int(name[-1])+idset)+'/fw_'+'_'+str(ntheta)+'/r', overwrite=True)
                res+=Tpsi
        np.save(name+'_4merged_bin' + str(binning)+str(kk)+'.npy',res)