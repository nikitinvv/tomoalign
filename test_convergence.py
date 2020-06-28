import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import tomoalign

if __name__ == "__main__":

    n = 256
    ntheta = 384
    pprot = 192
    adef = 10
    # tomoalign.gen_cyl_data(n, ntheta, pprot, adef)
    data = dxchange.read_tiff('data/deformed_data.tiff')
    theta = np.load('data/theta.npy')
    [ntheta, nz, n] = data.shape

    center = n/2
    pnz = n  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
    ptheta = 16  # number of projections for simultaneus processing by one GPU in the alignment sub-problem
    # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
    stepwin = 1
    ngpus = 4  # number of gpus
    nitercg = 64  # number of iterations in the cg scheme
    niteradmm = 256  # number of iterations in the ADMM scheme
    alpha = 1e-8  # regularization parameter

    for titer in [1,2,4,8]:        
        uof,lagrof = tomoalign.admm_of(data, theta, pnz, ptheta, center, stepwin, ngpus, niteradmm, titer)
        dxchange.write_tiff(uof, 'data/of_recon/recon/iter'+str(niteradmm))
        np.save('data/of_recon/recon/lagr'+str(titer),lagrof)        

    plt.figure(figsize=(8,4))    
    for titer in [1,2,4,8]:        
        lagr = np.load('data/of_recon/recon/lagr'+str(titer)+'.npy')
        plt.plot(np.log(lagr[:64:4,3]),linewidth=0.5,label=str(titer))
        plt.grid()
        # plt.xlim([1,niteradmm//4])
        plt.legend(loc="upper right",fontsize=22)
        #plt.xticks(np.arange(0,1025,103),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=18)
        # plt.yticks(np.arange(0,1.1,0.2),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=18)
        # plt.xticks(np.arange(0,0.2,1.1),fontsize=16)
        # plt.arrow(200, 0.9, -127, -0.65, color='gray',width=0.02, head_width=0) 
        # plt.arrow(200, 0.7, -72, -0.48, color='gray',width=0.02, head_width=0) 


        plt.savefig('data/of_recon/convergence_inner_iter.png')
