

from .smn4_album import Album
from .smn4_loader import FmriLoader
from .concatenater import Concatenater
from .ridge_regression import RidgeModel

from scipy.io import loadmat
from scipy.stats import zscore
from os.path import join
from functools import reduce
from .smn4_utils import *
import numpy as np
from tqdm import tqdm
import pickle

class WordRateModel(Album):
    '''
        This word rate model is generated from 'Huth etc 2022 Method-word_rate_model', used to predict word rate of fMRI

        sub: subject
        tr_list: list of tr delays. For example, tr_list = [1,2,3], then we concatenate the response 
            from (t+1, t+2, t+3) to predict the word rate at time t.
    '''
    def __init__(self, ridge_config, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ridge_config = ridge_config
        self.fmri_loader = FmriLoader(**kwargs)
        self.ridge_model = RidgeModel(**ridge_config, )
        self.catter = Concatenater()
        
    def run(self, sub, tr_list, word_rates, ):
        '''
            cat fmri from tr_list, and return to Weight: fMRI_cat_features * Weight = Word_rate
        '''
        
        fmri_loader = self.fmri_loader
        word_rate_path = get_word_rate_model_path(sub, tr_list, self.result_path, **self.ridge_config)
        # cat fmri
        fmri_data_cats = np.array([])
        for i in tqdm(self.story_range, desc=f"time_delays = {[tr * self.fmri_tr for tr in tr_list]}, concatenating..."):
            # get fMRI data of each story
            fmri_data = fmri_loader.load_one(sub, story=i)
            fmri_data_cat = self.catter.cat(fmri_data, tr_list)

            fmri_data_cats = np.concatenate((fmri_data_cats, fmri_data_cat), axis=0) if fmri_data_cats.any() \
                else fmri_data_cat

        assert fmri_data_cats.shape[0] == word_rates.shape[0], \
            f"fmri_datas.shape = {fmri_data_cats.shape}, word_rates.shape = {word_rates.shape}"
        
        # regression
        ridge_model = self.ridge_model
        self.log.info(f"Training word rate model, X.shape = {fmri_data_cats.shape}, y.shape = {word_rates.shape}")
        ridge_cv, avg_corrs = ridge_model.ridge(X=fmri_data_cats, y=word_rates)
        
        self.log.info(f"{ridge_cv}, {avg_corrs}")
        with open(word_rate_path, "wb") as f:
            pickle.dump({"ridge_cv": ridge_cv, "avg_corrs": avg_corrs}, f)
        
        return 

class WordRateCounter(Album):
    '''
        This model is used to count the word rate of stimuli(ground truth), and return to an np.array (length: fMRI_tr)
        two method is considered: one is counted as the paper, the other is using convolving(this method hasn't been inplemented.).
        Params:
            word_rate_method: (now we only have 'count_delays')
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def count(self, **kwargs):
        '''
            Return: list of array, each array is the word rate of a story.
        '''
        if self.word_rate_method == 'count_delays':
            return self.count_delays()
        else:
            return None

    def count_delays(self, ):
        time_counts = np.array([])
        for i in self.story_range:
            time_count = self.count_delays_one(i, )
            time_counts = np.concatenate((time_counts, time_count), )
        return time_counts

    def count_delays_one(self, story, is_zscored=True):
        '''
            use `count_delay` on one story. 

            Arg:
                is_zscored: results will be z-scored if True
        '''
        feature_abandon = self.feature_abandon
        time = loadmat(join(self.time_path, f'story_{story}.mat'))
        start_time = np.squeeze(time['start'])
        start_time = np.round(start_time * 100).astype('int') # use rint to avoid some unexpected error as: int(1.13*100) = 112
        end_time = np.squeeze(time['end'])
        end_time = np.round(end_time * 100).astype('int') 

        length = end_time[-1]
        word_exist_times = np.zeros((length, )) # 0.01s

        for t in end_time:
            word_exist_times[t - 1] += 1

        time_count = []
        for tr in range(0, length, 71):
            time_count.append(round(np.sum(word_exist_times[tr: tr + 71])))

        # attention: because of this feature_abandon, word rate of the start (about 3 tr) is deleted.
        time_count = np.array(time_count[feature_abandon: self.ref_length[story - 1] + feature_abandon])

        # z-scored
        time_count = zscore(time_count) if is_zscored \
            else time_count

        return time_count

