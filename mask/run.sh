# CUDA_VISIBLE_DEVICES=0,1,2,3 python admm.py 4 200 /local/data/vnikitin/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 python admm.py 8 200 /local/data/vnikitin/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002 &
#CUDA_VISIBLE_DEVICES=0,1,2,3 python admm.py 15 200 /local/data/vnikitin/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002 &
#CUDA_VISIBLE_DEVICES=4,5,6,7 python admm.py 2 200 /local/data/vnikitin/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002/Chip_9100eV_interlaced_200prj_per_rot_3000prj_1s_002 &
CUDA_VISIBLE_DEVICES=4,5,6,7 python admm.py 1 720 /local/data/vnikitin/mask/Sample9/Sple9_interlaced_721prj_1s_127
CUDA_VISIBLE_DEVICES=4,5,6,7 python admm.py 1 720 /local/data/vnikitin/mask/Sample11/Sple11_flyscan_721prj_0.5s_132
CUDA_VISIBLE_DEVICES=4,5,6,7 python admm.py 1 720 /local/data/vnikitin/mask/Sample15/Sple15_flyscan_721prj_0.5s_136