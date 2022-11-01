# used for feature convolver
# TIME of Word-time pairs and of fMRI-time pairs are not aligned. This model is used to solve this problem.

import scipy.io as scio
import h5py
import hdf5storage as hdf5
import numpy as np
from nilearn import glm
import os
from os.path import join
from tqdm import tqdm
from album import Album

zs = lambda v: (v-v.mean())/v.std()

class FeatureConvolver(Album):
    def __init__(self):
        super().__init__()
        self.hrf = glm.first_level.spm_hrf(self.fmri_tr, self.fmri_tr_int)

    def forward(self, if_save=True, **kwargs):
        '''
            in_path: folder of embeddings
            time_path: folder of saved time points
            out_path: folder of results
            ref_length: 
            convolve_type: convolve method we take
        '''

        convolve_type = self.convolve_type
        layer = self.layer_num
        time_type = self.duration_time_type
        story_num = self.n_story
        ref_length = self.ref_length
        FMRI_TR_INT = self.fmri_tr_int

        log = self.log
        for i in tqdm(range(1, story_num + 1)):
            log.info(f"start story_{i} convolving...")
            # load story and fmri
            data = scio.loadmat(join(self.embedding_path, f'derivatives:annotations:embeddings:gpt:word-level:story_{i}_word_gpt_0-24_1024.mat'))
            data = data['data'][layer] # text_len * dim

            time = scio.loadmat(join(self.time_path, f'derivatives:annotations:time_align:word-level:story_{i}_word_time.mat'))
            start_time = np.squeeze(time['start'])
            start_time = np.round(start_time * 100).astype('int') # use rint to avoid some unexpected error as: int(1.13*100) = 112
            end_time = np.squeeze(time['end'])
            end_time = np.round(end_time * 100).astype('int') 

            assert data.shape[0] == start_time.shape[0], \
                f"data and tr not suitable, data_num = {data.shape[1]}, while tr_num = {start_time.shape[0]}"

            tr_limit, dim = data.shape[0], data.shape[1]

            # upsampling stimuli to 0.01s
            if convolve_type == 'offset':
                length = end_time[-1]
                time_series = np.zeros([length, dim])
                t = 0
                for j in range(length):
                    if j == end_time[t]:
                        time_series[j] = data[t]
                        while j == end_time[t]:
                            t += 1
                            if t == tr_limit:
                                break

            elif convolve_type == 'duration':
                if time_type == 'start':
                    word_time = start_time
                else:
                    word_time = end_time
                length = word_time[-1]
                time_series = np.zeros([length + FMRI_TR_INT, dim])
                t = 0
                for j in range(length):
                    if j == word_time[t]:
                        for k in range(FMRI_TR_INT):
                            time_series[j + k] += data[t]
                        while j == word_time[t]:
                            t += 1
                            if t == tr_limit:
                                break

            elif convolve_type == 'seduration': 
                '''
                    duration between start and end of each word
                '''
                length = start_time[-1]
                time_series = np.zeros([length, dim])
                t = 0
                for j in range(length):
                    if j == start_time[t]:
                        for k in range(j, end_time[t]):
                            time_series[k] += data[t]                            
                        while j == start_time[t]:
                            t += 1
                            if t == tr_limit:
                                break

            # Downsampling to 0.71s
            tr_length = ref_length[i - 1]
            conv_series_ds = np.zeros([tr_length, dim])
            abandon = self.feature_abandon

            for d in range(dim):
                tmp_series = np.squeeze(time_series[:, d])

                conv_series = np.convolve(self.hrf, tmp_series)
                conv_series = conv_series[: length]
                conv_series_tmp = [conv_series[j] for j in range(0, length, FMRI_TR_INT)]
                conv_series_tmp = np.array(conv_series_tmp)
                word_feature = zs(conv_series_tmp[abandon: tr_length + abandon])
                
                conv_series_ds[:, d] += np.squeeze(word_feature)

                # logging.info(conv_series_ds[:, d])

            if if_save:
                hdf5.writes({"word_time_pair": conv_series_ds.astype('float32')}, \
                    join(self.result_path, f'story_{i}.mat'), matlab_compatible=True)

if __name__ == "__main__":
    feature_convolver = FeatureConvolver()
    feature_convolver(if_save=False)