#!/bin/bash
source /home/vvnikitin/anaconda3/etc/profile.d/conda.sh
conda activate tomoalign
#NUMEXPR_MAX_THREADS=128 python processing.py /lus/theta-fs0/projects/nanoct/Lethien/Sample1_16nmZP_8keV_2200prj_219.h5
#NUMEXPR_MAX_THREADS=128 python processing.py /lus/theta-fs0/projects/nanoct/Lethien/Sample2_16nmZP_8keV_1400prj_220.h5  
#NUMEXPR_MAX_THREADS=128 python processing.py /lus/theta-fs0/projects/nanoct/Lethien/Sample3_16nmZP_8keV_3000prj_221.h5  
#NUMEXPR_MAX_THREADS=128 python processing.py /lus/theta-fs0/projects/nanoct/Lethien/Sample4_16nmZP_8keV_2200prj_222.h5 

if [[ $1 -eq 1 ]]
then
       NUMEXPR_MAX_THREADS=128 python admm.py /lus/theta-fs0/projects/nanoct/Lethien/Sample1_16nmZP_8keV_2200prj_219.h5
       NUMEXPR_MAX_THREADS=128 python cg.py /lus/theta-fs0/projects/nanoct/Lethien/Sample1_16nmZP_8keV_2200prj_219.h5
fi

if [[ $1 -eq 2 ]]
then
	NUMEXPR_MAX_THREADS=128 python admm.py /lus/theta-fs0/projects/nanoct/Lethien/Sample2_16nmZP_8keV_1400prj_220.h5  
	NUMEXPR_MAX_THREADS=128 python cg.py /lus/theta-fs0/projects/nanoct/Lethien/Sample2_16nmZP_8keV_1400prj_220.h5  
fi

if [[ $1 -eq 3 ]]
then
	NUMEXPR_MAX_THREADS=128 python admm.py /lus/theta-fs0/projects/nanoct/Lethien/Sample3_16nmZP_8keV_3000prj_221.h5  
	NUMEXPR_MAX_THREADS=128 python cg.py /lus/theta-fs0/projects/nanoct/Lethien/Sample3_16nmZP_8keV_3000prj_221.h5  
fi

if [[ $1 -eq 4 ]]
then
	NUMEXPR_MAX_THREADS=128 python admm.py /lus/theta-fs0/projects/nanoct/Lethien/Sample4_16nmZP_8keV_2200prj_222.h5 
	NUMEXPR_MAX_THREADS=128 python cg.py /lus/theta-fs0/projects/nanoct/Lethien/Sample4_16nmZP_8keV_2200prj_222.h5 
fi

