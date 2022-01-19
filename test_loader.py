import h5py
import numpy as np
f = h5py.File('/home/lixiangpeng/data/dataset/videocaption/msrvtt_s3d_new/MSRVTT/vid_feat_files/mult_h5/v/i/d/video523.h5','r')
for group in f.keys():
    print(group)
    group_read = f[group]
    for subgroup in group_read:
        print(subgroup)
print(f.keys())
