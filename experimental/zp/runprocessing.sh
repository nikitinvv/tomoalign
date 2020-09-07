#!/usr/bin/bash
#SBATCH -n 40
#SBATCH --mem 200G
#SBATCH -t 40:00:00
cd /mxn/visitors/vviknik/tomoalign_vincent/tomoalign_develop2/experimental/zp
# python processing.py 24 200 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/DeAndrade/16nmZP/Kenan_ZP_ROI2_8keV_interlaced_5000prj_3s_002
python processing.py 24 200 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/DeAndrade/16nmZP/Kenan_ZP_8keV_interlaced_5000prj_3s_001