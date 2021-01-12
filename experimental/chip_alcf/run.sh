#!/bin/bash
source /home/vvnikitin/anaconda3/etc/profile.d/conda.sh
conda activate tomoalign
#python processing.py 1 3900 0 /lus/theta-fs0/projects/nanoct/chip_16nmZP_tube_lens_interlaced_4000prj_3s_094
python admm94.py 1 1 3900 0 /lus/theta-fs0/projects/nanoct/chip_16nmZP_tube_lens_interlaced_4000prj_3s_094
