import h5py
import numpy as np

mat_path = 'story_1_pku_fmri.mat'
mat = h5py.File(mat_path)
x = mat.keys()
print(x)
mat = np.transpose(mat['fmri_response'])
print(mat)
print(mat.shape)