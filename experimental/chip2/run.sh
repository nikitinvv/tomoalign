#!/usr/bin/bash
#SBATCH -n 64
#SBATCH -p v100
#SBATCH --exclude gn1
#SBATCH --mem 160G
#SBATCH -t 20:00:00
nvidia-smi
module add GCC/8.3.0 iccifort/2019.5.281 CUDA/10.1.243
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign/chip2
#python admm.py 3 1000 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073_2merged
python admm.py 3 1000 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073;
#python admm.py 3 1000 2 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073;
 #python admm.py 3 1000 3 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073;python admm.py 3 1000 4 /data/staff/tomograms/vviknik/tomoalign_vincent_data/chipMay/Chip_ZP_16nmZP_9100eV_interlaced_3000prj_stabiliz_2s_3s_073;