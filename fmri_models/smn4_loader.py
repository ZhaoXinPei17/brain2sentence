
import torch
from tqdm import tqdm
from os.path import join
import h5py
from scipy.io import loadmat
import numpy as np

from .smn4_utils import *

class Loader():

    @staticmethod
    def get_voxel_top_index(voxel_top_path: str, head: str, voxel_top_num, fmri_voxel_num):
        '''
            Arg:
                voxel_top_path: used for load voxel_top_index
                voxel_top_num: if None, use all voxels.
                fmri_voxel_num: full num of voxels
            Return: 
                voxel_index: index of voxel top k
        '''       
        if voxel_top_num is None:
            voxel_index = np.array(range(fmri_voxel_num))
        else:
            try:
                score = loadmat(voxel_top_path)
                score = np.squeeze(score[head])
                voxel_index = np.argsort(score)
                voxel_index = voxel_index[-voxel_top_num: ]
        
            except FileNotFoundError as e:
                # print(repr(e))
                voxel_index = np.array(range(voxel_top_num))

        return voxel_index

class FmriLoader(Loader):
    '''
        Load fmri from smn4 datasets.
        voxel_top_num: is None if not voxel_top
    '''
    def __init__(self, 
        result_path, 
        voxel_top_num,
        fmri_path,
        test_id, 
        ref_length,
        n_story, 
        **fmri_config) -> None:
        
        self.result_path = result_path
        self.voxel_top_num = voxel_top_num
        self.fmri_path = fmri_path
        self.test_id = test_id
        self.ref_length = ref_length
        self.n_story = n_story
        self.story_range = get_story_range(n_story)

    def load(self, sub):
        '''
            Return:
            train_fmri: (n_TRs, n_voxels); 
            starts: (n_story + 1, ), the start index of each story in the fmri_path. 
        '''
        train_fmri = torch.tensor([])
        starts = [0]

        for i in tqdm(self.story_range, desc=f'loading fmri from {self.fmri_path} ...'):
            if i == self.test_id:
                continue
            
            single_fmri = self.load_one(sub, story=i)

            train_fmri = torch.cat([train_fmri, single_fmri])
            starts.append(train_fmri.shape[0])

        assert train_fmri.shape[0] == sum(self.ref_length[ : self.n_story]), \
            f"Unmatch: train_fmri_tr = {train_fmri.shape[0]}, reference_tr = {sum(self.ref_length[ : self.n_story])}"
        
        return train_fmri, starts

    def load_one(self, sub, story: int):
        '''
            Return: train_fmri: (n_TRs, n_voxels); 
        '''
        if story < 0:
            return None

        fmri_file = join(self.fmri_path, sub, f'story_{story}.mat')
        data = h5py.File(fmri_file, 'r')
        single_fmri = np.array(data['fmri_response'])
        
        # voxel top
        voxel_top_path = get_voxel_top_path(sub, self.result_path)
        voxel_index = self.get_voxel_top_index(voxel_top_path, 'test_corrs', self.voxel_top_num, single_fmri.shape[1])
        single_fmri = single_fmri[:, voxel_index]

        return torch.FloatTensor(single_fmri)

class FeatureLoader(Loader):
    '''
        Load feature from smn4 datasets.
    '''
    def __init__(self, 
        feature_path, 
        feature_type,
        test_id, 
        n_story, 
        **feature_config) -> None:
        
        self.feature_dir = feature_path
        self.feature_type = feature_type
        self.test_id = test_id
        # self.n_story = n_story
        self.story_range = get_story_range(n_story)

    def load(self, ):
        '''
            Return:
            train_feature: (n_TRs, feature_dim)
        '''
        train_feature = torch.tensor([])
        starts = [0]
        for i in tqdm(self.story_range, desc=f'loading stimulus from {self.feature_dir} ...'):
            if i == self.test_id:
                continue
            
            single_feature = self.load_one(i)
            train_feature = torch.cat([train_feature, single_feature])
            starts.append(train_feature.shape[0])
                            
        return train_feature, starts

    def load_one(self, story_ind, ):
        '''
            Return:
                torch.tensor, dtype=float32
        '''
        if story_ind < 0:
            return None
        feature_path = get_feature_path(story_ind, self.feature_dir)
        data = h5py.File(feature_path, 'r')
        return torch.FloatTensor(np.array(data[self.feature_type])).transpose(0, 1)

class WeightLoader(Loader):
    def __init__(self, 
        result_path, 
        voxel_top_num, 
        **kwargs):

        self.result_path = result_path
        self.voxel_top_num = voxel_top_num

    def load(self, sub):
        encoding_weights_path = get_weight_path(sub, self.result_path)
        encoding_weights = loadmat(encoding_weights_path)["weights"]

        voxel_top_path = get_voxel_top_path(sub, self.result_path)
        voxel_index = self.get_voxel_top_index(voxel_top_path, 'test_corrs', self.voxel_top_num)

        return encoding_weights[:, voxel_index]
