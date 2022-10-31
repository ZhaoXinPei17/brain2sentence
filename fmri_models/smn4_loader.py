
import torch
from tqdm import tqdm
from os.path import join
import h5py
import scipy.io as scio
import numpy as np

from smn4_album import Album

class FmriLoader(Album):
    '''
        Load fmri from smn4 datasets.
        voxel_top_num: is None if not voxel_top
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, sub):
        return self.load(sub=sub)

    def load(self, sub):
        '''
            Return:
            train_fmri: (n_TRs, n_voxels); 
            starts: (n_story + 1, ), the start index of each story in the fmri_path. 
        '''
        voxel_top_path = (self.fmri_feature_corrs_path).replace("$$", sub)
        train_fmri = torch.tensor([])
        starts = [0]

        if self.voxel_top_num is not None:
            score = scio.loadmat(voxel_top_path)
            score = np.squeeze(score['test_corrs'])
            # print(score.shape)

            voxel_index = np.argsort(score)
            # print(voxel_index.shape)
            voxel_index = voxel_index[-self.voxel_top_num: ]
            # print(voxel_index.shape, voxel_index[:10])

        for i in tqdm(self.story_range, desc=f'loading fmri from {self.fmri_path} ...'):
            if i == self.test_id:
                continue
            fmri_file = join(self.fmri_path, sub, f'story_{i}.mat')
            data = h5py.File(fmri_file, 'r')

            single_fmri = np.array(data['fmri_response'])
            if self.voxel_top_num is not None:
                # voxel top
                single_fmri = single_fmri[:, voxel_index]

            train_fmri = torch.cat([train_fmri, torch.FloatTensor(single_fmri)])
            starts.append(train_fmri.shape[0])
            # print(f"story = {i}, single_fmri_shape = {single_fmri.shape}, train_fmri_shape = {train_fmri.shape}")

        assert train_fmri.shape[0] == sum(self.ref_length[ : self.n_story]), \
            f"Unmatch: train_fmri_tr = {train_fmri.shape[0]}, reference_tr = {sum(self.ref_length[ : self.n_story])}"
        
        return train_fmri, starts

    def load_one(self, sub, story: int):
        '''
            Return: train_fmri: (n_TRs, n_voxels); 
        '''
        voxel_top_path = (self.fmri_feature_corrs_path).replace("$$", sub)
        
        if self.voxel_top_num is not None:
            score = scio.loadmat(voxel_top_path)
            score = np.squeeze(score['test_corrs'])
            voxel_index = np.argsort(score)
            voxel_index = voxel_index[-self.voxel_top_num: ]
            # print(voxel_index.shape, voxel_index[:10])
        
        fmri_file = join(self.fmri_path, sub, f'story_{story}.mat')
        data = h5py.File(fmri_file, 'r')

        single_fmri = np.array(data['fmri_response'])
        if self.voxel_top_num is not None:
                # voxel top
            single_fmri = single_fmri[:, voxel_index]

        return torch.FloatTensor(single_fmri)

class FeatureLoader(Album):
    '''
        Load feature from smn4 datasets.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, sub):
        return self.load(sub=sub)

    def load(self, ):
        '''
            Return:
            train_feature: (n_TRs, feature_dim)
        '''
        train_feature = torch.tensor([])
        feature_type = self.feature_type
        starts = [0]
        for i in tqdm(self.story_range, desc=f'loading stimulus from {self.feature_path} ...'):
            if i == self.test_id:
                continue
            feature_file = join(self.feature_path, f'story_{i}.mat')
                
            data = h5py.File(feature_file, 'r')
            single_feature = torch.tensor(np.array(data[feature_type])).transpose(0, 1)
            train_feature = torch.cat([train_feature, single_feature])
            starts.append(train_feature.shape[0])
                
            # self.log.info(f"story = {i}, single_feature_tr = {single_feature.shape[0]}, train_feature_tr = {train_feature.shape[0]}")
            
        return train_feature, starts

if __name__ == '__main__':
    
    fmri_loader = FmriLoader(voxel_top_num=None)
    print((fmri_loader.load_one(sub='02', story=1)).shape)
