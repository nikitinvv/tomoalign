#!/usr/bin/bash
#SBATCH -n 64
#SBATCH -p v100
#SBATCH --exclude gn1
#SBATCH --mem 164G
#SBATCH -t 40:00:00
nvidia-smi
module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign/brain2
#python admm.py 3 1000 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073
#python cg.py 4 720 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167;
# python admm.py 4 720 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167;
python admm_shift.py 4 720 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167;
# python cg.py 1 720 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167;
# python cg.py 2 720 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_2880prj_1440deg_167;
#python admm_shift.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_4800prj_720deg_166;
#python admm_shift.py 2 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_4800prj_720deg_166;
#python admm.py 4 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_4800prj_720deg_166;
#python admm_shift.py 4 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/brain/Brain_Petrapoxy_day2_4800prj_720deg_166;

