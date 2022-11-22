import os
from os.path import join as pjoin
import pickle
## Calculate

def mult_diag(d, mtx, left=True):
    """
        the code is adapted from https://github.com/HuthLab/speechmodeltutorial
        Multiply a full matrix by a diagonal matrix.
        This function should always be faster than dot.
        Input:
            d -- 1D (N,) array (contains the diagonal elements)
            mtx -- 2D (N,N) array

        Output:
            mult_diag(d, mts, left=True) == dot(diag(d), mtx)
            mult_diag(d, mts, left=False) == dot(mtx, diag(d))
        
        By Pietro Berkes
        From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    # return (d*mtx.T).T     
    return (d*mtx.transpose()).transpose() if left else d*mtx

## Paths

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_time_path(story_index: int, time_path, **kwargs):
    '''
        Arg:
            album: Album of configs
            story_index
    '''
    return pjoin(time_path, f"story_{story_index}.mat")

def get_word_rate_model_path(sub: str, tr_list, result_path, encoding_method, **kwargs):
    sub_dir = pjoin(result_path, f"sub_{sub}")
    mkdir(sub_dir)
    return pjoin(sub_dir, f"word_rate_model--{encoding_method}--{tr_list}")

def get_feature_path(story_ind: int, feature_path):
    return pjoin(feature_path, f"story_{story_ind}.mat")

def loadcls(path, ):
    with open(path, "rb") as f:
        cls = pickle.load(f)
    return cls

def savecls(path, cls):
    with open(path, "wb") as f:
        pickle.dump(cls, f)

def get_cls_path(sub, result_path, t_list):
    '''
        tfold: if we make our regression as some parts, `tfold` is used for justifying each part.
    '''
    sub_dir = pjoin(result_path, f"sub_{sub}"); mkdir(sub_dir)
    return pjoin(sub_dir, f"weight--sub_{sub}--time_{t_list}")



def get_voxel_top_path(sub: str, result_path, ):
    return get_avg_corrs_path(sub, result_path, 'all')
    # need to judge



def get_corrs_path(sub, result_path, t_list):
    '''
        get corrs path for subject.
    '''
    sub_dir = pjoin(result_path, f"sub_{sub}"); mkdir(sub_dir)
    return pjoin(sub_dir, f"sub_{sub}--time_{t_list}.mat")



def get_avg_corrs_path(sub, result_path, t_list):
    
    sub_dir = pjoin(result_path, f"sub_{sub}"); mkdir(sub_dir)
    return pjoin(sub_dir, f"sub_{sub}--time_{t_list}--average.mat")



def get_nii_path(sub, result_path, feature_type, method, t_list):

    sub_dir = pjoin(result_path, f"sub_{sub}"); mkdir(sub_dir)
    return pjoin(sub_dir, f"sub_{sub}--{feature_type}--{method}--time_{t_list}.dscalar.nii")




## general utils

def get_dim(shape):
    '''
        get dim of a 2-dimension Tensor
        Arg:
            shape
    '''
    try:
        dim = shape[1]
    except:
        dim = 1 

    return dim

def get_story_range(n_story):
    return range(1, n_story + 1) 