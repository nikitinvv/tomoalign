#!/usr/bin/bash
#SBATCH -n 64
#SBATCH -p v100
##SBATCH --exclude gn1
#SBATCH --mem 164G
#SBATCH -t 40:00:00
nvidia-smi
module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign/nmc
# python cg.py 1 1210 /data/staff/tomograms/vviknik/tomoalign_vincent_data/nmc/811-8-1_8400eV_405
# python cg.py 1 1210 /data/staff/tomograms/vviknik/tomoalign_vincent_data/nmc/811-8-1_8320eV_406
# python admm.py 1 1210 /data/staff/tomograms/vviknik/tomoalign_vincent_data/nmc/811-8-1_8400eV_405
python admm.py 1 1210 /data/staff/tomograms/vviknik/tomoalign_vincent_data/nmc//811-8-1_8320eV_406