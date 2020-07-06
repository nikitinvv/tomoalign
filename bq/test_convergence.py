import matplotlib.pyplot as plt
import numpy as np
import dxchange
import scipy.ndimage as ndimage
import matplotlib
import tomoalign
plt.rc('text', usetex=True)
# matplotlib.rc('font', family='serif', serif='cm10')
# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']


if __name__ == "__main__":

    n = 256
    ntheta = 384
    adef = 10
    noise = False

    for pprot in [96,192,384]:
        namepart = '_pprot'+str(pprot)+'_noise'+str(noise)
        tomoalign.gen_cyl_data(n, ntheta, pprot, adef, noise)
        # exit()
        data = dxchange.read_tiff('data/deformed_data'+namepart+'.tiff')
        theta = np.load('data/theta'+namepart+'.npy')
        [ntheta, nz, n] = data.shape

        center = n/2
        pnz = 64  # number of slice for simultaneus processing by one GPU in the tomography sub-problem
        ptheta = 32  # number of projections for simultaneus processing by one GPU in the alignment sub-problem
        # step for decreasing window size (increase resolution) in Farneback's algorithm on each ADMM iteration
        stepwin = 1
        ngpus = 4  # number of gpus
        niteradmm = 256  # number of iterations in the ADMM scheme
        
        
        for titer in [1,2,4,8,16,32]:        
            uof,lagrof = tomoalign.admm_of(data, theta, pnz, ptheta, center, stepwin*titer, ngpus, niteradmm//titer+1, titer)
            dxchange.write_tiff(uof, 'data/of_recon/'+namepart+'/recon/iter'+str(niteradmm))
            np.save('data/of_recon/lagr'+namepart+str(titer),lagrof)        

        fig = plt.figure(figsize=(8,5))    
        ax = fig.add_subplot(111)
        for titer in [1,2,4,8,16,32]:        
            print(titer)
            lagr = np.load('data/of_recon/lagr'+namepart+str(titer)+'.npy')
            plt.plot(1+np.arange(niteradmm//titer+1)*titer,lagr[:,3],linewidth=1.5,label='('+str(niteradmm//titer)+'/'+str(titer)+')')
            plt.xlim([0,niteradmm])
            plt.legend(title='(ADMM/inner) iterations', loc="upper right",fontsize=20)
            ax.get_legend().get_title().set_fontsize('20')
            plt.xlabel('joint iteration',fontsize=22)
            plt.ylabel('error', rotation=90, fontsize=22)
            plt.grid(True)
            plt.xticks([0, 64, 128,192, 256])#,['0', r'64', r'128', r'192', r'256'])
            # plt.yticks(fontname = "Times New Roman") 
            
            #plt.xticks(np.arange(0,1025,103),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=20)
            # plt.yticks(np.arange(0,1.1,0.2),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=20)
            # plt.xticks(np.arange(0,0.2,1.1),fontsize=20)
            # plt.arrow(200, 0.9, -127, -0.65, color='gray',width=0.02, head_width=0) 
            # plt.arrow(200, 0.7, -72, -0.48, color='gray',width=0.02, head_width=0) 
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)


        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('data/of_recon/convergence_inner_iter'+namepart+'.png')

    
        fig = plt.figure(figsize=(8,4))    
        ax = fig.add_subplot(111)
        titer = 1
        lagr = np.load('data/re_recon/lagr'+namepart+str(titer)+'.npy')
        plt.plot(1+np.arange(niteradmm//1+1)*titer,lagr[:,3],linewidth=1.5,label='Re-projection')
        titer = 4
        lagr = np.load('data/of_recon/lagr'+namepart+str(titer)+'.npy')
        plt.plot(1+np.arange(niteradmm//4+1)*titer,lagr[:,3],linewidth=1.5,label='ADMM')
        
        plt.xlim([0,niteradmm])
        plt.legend(loc="upper right",fontsize=18)
        ax.get_legend().get_title().set_fontsize('18')
        plt.xlabel('joint iteration',fontsize=18)
        plt.ylabel('error', rotation=90, fontsize=18)
        plt.grid(True)
        plt.xticks([0, 64, 128,192, 256])#,['0', r'64', r'128', r'192', r'256'])
            # plt.yticks(fontname = "Times New Roman") 
            
            #plt.xticks(np.arange(0,1025,103),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=18)
            # plt.yticks(np.arange(0,1.1,0.2),[r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}',r'\textbf{1.0}'],fontsize=18)
            # plt.xticks(np.arange(0,0.2,1.1),fontsize=18)
            # plt.arrow(200, 0.9, -127, -0.65, color='gray',width=0.02, head_width=0) 
            # plt.arrow(200, 0.7, -72, -0.48, color='gray',width=0.02, head_width=0) 
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)

        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('data/re_recon/convergence_inner_iter'+namepart+'.png', dpi=300)

    