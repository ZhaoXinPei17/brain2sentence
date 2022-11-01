
# import logging
# LOGGING_FILE = "/sdb/xpzhao/zxps/brain2char/logs/word_rate_log"
# logging.basicConfig(filename= LOGGING_FILE, format= '%(asctime)s %(message)s', datefmt= '%m/%d/%Y %H:%M')
import sys
sys.path.append(f"/sdb/xpzhao/zxps") 

from tqdm import tqdm
from argparse import ArgumentParser
from utils import FmriLoader, Album
import torch
import yaml
import scipy.io as scio
import h5py
from os.path import join
from functools import reduce
import numpy as np
from sklearn import linear_model

zs = lambda v: (v-v.mean(0))/v.std(0)

class WordRateModel(Album):
    '''
        This word rate model is generated from 'Huth etc 2022 Method-word_rate_model'.

        tr_list: list of tr delays. For example, tr_list = [1,2,3], then we concatenate the response 
            from (t+1, t+2, t+3) to predict the word rate at time t.
    '''
    def __init__(self, sub, tr_list, **kwargs) -> None:
        super().__init__()
        self.sub = sub
        self.tr_list = tr_list

    def ridge_delay(self, word_rates):
        '''
            cat fmri from tr_list, and return to Weight: fMRI_cat_features * Weight = Word_rate
        '''
        
        fmri_loader = FmriLoader(self.sub)
        fmri_datas, starts = fmri_loader.load()

        # cat fmri
        tr_max = self.tr_list[-1]
        fmri_data_cats = []
        for i in tqdm(self.story_range, desc=f"time_delays = {[tr*self.fmri_tr for tr in self.tr_list]}, concatenating..."):
            fmri_data = fmri_datas[starts[i-1]:starts[i], :]
            fmri_data_cat = fmri_data[self.tr_list[0]:-tr_max+self.tr_list[0], :]

            for tr in self.tr_list[1: ]:
                if -tr_max+tr < 0:
                    fmri_data_tmp = fmri_data[ tr:-tr_max+tr , :]
                elif -tr_max+tr == 0: # avoid fmri_data[tr:0]
                    fmri_data_tmp = fmri_data[ tr: , :]
                fmri_data_cat = np.concatenate((fmri_data_cat, fmri_data_tmp), axis=1)

            # print(fmri_data_cat.shape)
            fmri_data_cats.append(fmri_data_cat)

        # cat fmri data of each story
        fmri_train_data = reduce(lambda x, y: np.concatenate((x, y), axis=0), fmri_data_cats)
        # cat word rate, and remember to remove the last tr of each word_rate array
        word_rate_train = (reduce(lambda x, y: np.concatenate((x[:-tr_max], y), axis=0), word_rates))[:-tr_max]
        # print(fmri_train_data.shape, word_rate_train.shape)

        print(f"fmri_datas.shape = {fmri_train_data.shape}, word_rates.shape = {word_rate_train.shape}. training word_rate_model...")
        reg = linear_model.Ridge(alpha= 1., )
        reg.fit(fmri_train_data, word_rate_train, )
        # print(reg.coef_)

        return 


class WordRateCounter(Album):
    '''
        generated from feature_convolver.
        This is used to count the word rate of stimuli, and return to an np.array (length: fMRI_tr)
        two method is considered: one is counted as the paper, the other is using convolving.
    '''

    def __init__(self):
        super().__init__()
        
    def count(self, **kwargs):
        '''
            Return: list of array, each array is the word rate of a story.
        '''
        if self.word_rate_method == 'count_delays':
            return self.count_delays()

    def count_delays(self, ):
        feature_abandon = self.feature_abandon

        time_counts = []
        for i in self.story_range:
            time = scio.loadmat(join(self.time_path, f'story_{i}.mat'))
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
                time_count.append(np.sum(word_exist_times[tr: tr + 71]))

            time_count = time_count[feature_abandon: self.ref_length[i - 1] + feature_abandon]
            # attention: because of this feature_abandon, word rate of the start (about 3 tr) is deleted.

            time_count = np.array(time_count)
            # print(time_count.shape)
            time_counts.append(time_count)
        return time_counts


if __name__ == '__main__':

    word_rate_counter = WordRateCounter()
    word_rate_model = WordRateModel('01', [2,4,6,8,10,12])

    word_rates = word_rate_counter.count()
    word_rate_model.ridge_delay(word_rates)

