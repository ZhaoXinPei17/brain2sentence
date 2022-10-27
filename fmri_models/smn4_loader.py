
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
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, sub):
        '''
            Return:
            train_fmri: (n_TRs, n_voxels); 
            starts: list, the start index of each story in the fmri_path. len = n_story + 1
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


if __name__ == '__main__':
    
    fmri_loader = FmriLoader()
    print((fmri_loader.load_one(sub='02', story=1)).shape)
