#!/usr/bin/bash
#SBATCH -n 40
#SBATCH -p v100
##SBATCH --exclude gn1
#SBATCH --mem 160G
#SBATCH -t 40:00:00
# module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign_develop2/experimental/sple
python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple1_Phase_1201prj_interlaced_1s_010
python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple1_Phase_1201prj_interlaced_1s_010

python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple2_Phase_1201prj_interlaced_1s_011
python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple2_Phase_1201prj_interlaced_1s_011

# python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple3_Phase_1201prj_1s_009
# python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple3_Phase_1201prj_1s_009

# python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple4_Phase_1201prj_interlaced_1s_012
# python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple4_Phase_1201prj_interlaced_1s_012

# python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple5_Phase_1201prj_interlaced_1s_013
# python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple5_Phase_1201prj_interlaced_1s_013
