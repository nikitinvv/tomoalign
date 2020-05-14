import numpy as np 
import dxchange
# import deformcg as dc

# psi = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/tt/t3.tiff')[None,::2,::2].copy()
# data = dxchange.read_tiff('/data/staff/tomograms/vviknik/tomoalign_vincent_data/mask/Run4_9_1_40min_8keV_phase_100proj_per_rot_interlaced_1201prj_1s_024/tt/t3t.tiff')[None,::2,::2].copy()
# [ntheta,nz,n] = psi.shape
# mmin=[-0.25113034]
# mmax=[0.3569793]
# psi[psi<mmin[0]]=mmin[0]
# psi[psi>mmax[0]]=mmax[0]
# data[data<mmin[0]]=mmin[0]
# data[data>mmax[0]]=mmax[0]
# psi[:,1:]=data[:,:-1]
# pars = [0.5,5, 32, 4, 5, 1.1, 4]
# with dc.SolverDeform(1, nz, n, 1) as dslv:
#     print(np.linalg.norm(psi-data))
#     flow = dslv.registration_flow_batch(psi, data, mmin, mmax, None, pars, 1) 
#     Tpsi = dslv.apply_flow_gpu_batch(psi, flow)     
#     print(np.linalg.norm(Tpsi-data))
# 
# 
a=np.random.random(8)
#fa = np.fft.fftshift(np.fft.fft(np.fft.fftshift(a)))    
fa =np.fft.fft(a)
fae = np.zeros([16],dtype='complex128')
fae[4:-4]=fa
ae =np.fft.ifft(fae)*3

print(a)
print(ae)