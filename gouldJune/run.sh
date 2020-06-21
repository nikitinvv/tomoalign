#!/usr/bin/bash
#SBATCH -n 64
#SBATCH -p v100
#SBATCH --exclude gn1
#SBATCH --mem 160G
#SBATCH -t 80:00:00
nvidia-smi
module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign/gouldJune
python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/gouldJune/Sple1_laser_interlaced_721prj_0.5s_112