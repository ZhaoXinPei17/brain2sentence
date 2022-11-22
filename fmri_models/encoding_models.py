# init

from scipy.io import savemat
import torch
import numpy as np
import nibabel as nib
import logging
import pickle
from tqdm import tqdm
from functools import reduce

from .smn4_loader import FmriLoader, FeatureLoader
from .ridge_regression import RidgeCVModel
from .concatenater import Concatenater
from .smn4_utils import *

class EncodingBaseModel():
    """ 
        perform voxel-wise encoding 
    """
    def __init__(self, 
        sub,  
        alphas, 
        is_corrs_saved=False, 
        is_weights_saved=False, 

        **kwargs):

        test_id = kwargs.pop("test_id")
        result_path = kwargs.pop("result_path")
        voxel_top_num = kwargs.pop("voxel_top_num")
        fmri_path = kwargs.pop("fmri_path")
        ref_length = kwargs.pop("ref_length")
        n_story = kwargs.pop("n_story")
        feature_path = kwargs.pop("feature_path")
        feature_type = kwargs.pop("feature_type")
        encoding_method = kwargs.pop("encoding_method")
        brain_template = kwargs.pop("brain_template")
        fmri_voxel_num = kwargs.pop("fmri_voxel_num")
        block_shuffle = kwargs.pop("block_shuffle")
        nfold = kwargs.pop("nfold")
        inner_fold = kwargs.pop("inner_fold")
        train_ratio = kwargs.pop("train_ratio")
        test_ratio = kwargs.pop("test_ratio")
        plot_brain = kwargs.pop("plot_brain")

        self.fmri_loader = FmriLoader(
            result_path=result_path, 
            voxel_top_num=voxel_top_num,
            fmri_path=fmri_path,
            test_id=test_id,
            ref_length=ref_length,
            n_story=n_story,
            )

        self.feature_loader = FeatureLoader(
            feature_path=feature_path,
            feature_type=feature_type, 
            test_id=test_id,
            n_story=n_story, 
            )

        self.ridge_cv_model = RidgeCVModel(
            alphas=alphas,
            block_shuffle=block_shuffle,
            encoding_method=encoding_method,
            nfold=nfold,
            inner_fold=inner_fold,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            test_y=self.fmri_loader.load_one(sub=sub, story=test_id), 
            test_X=self.feature_loader.load_one(story_ind=test_id), 
            )

        self.catter = Concatenater()
        
        self.sub = sub
        
        self.result_path = result_path
        self.encoding_method = encoding_method
        self.brain_template = brain_template
        self.fmri_voxel_num = fmri_voxel_num
        self.is_corrs_saved = is_corrs_saved
        self.is_weights_saved = is_weights_saved
        self.plot_brain = plot_brain
        
        self.config = kwargs

    def plot_brain_corrs(self, corrs, t_list):
        base = nib.load(self.brain_template)
        try:
            corr_nii = nib.Cifti2Image(corrs.reshape(1, self.fmri_voxel_num), base.header)
        except Exception as e:
            print(repr(e))
            return False
        
        nii_name = get_nii_path(self.sub, self.result_path, self.feature_loader.feature_type, self.encoding_method, t_list)
        corr_nii.to_filename(nii_name)
        return True    
        
    def run_ridge(self, cat_tr_list, fMRI_fold):
        '''
            Arg:
                cat_tr_list: `List[int]`, used for concatenating word_features. 
                    We suggest concatenating word_features in 8 seconds before each fMRI.
                    for example: [1,2,4]: t+1, t+2, t+4
                fMRI_fold: `int`, used to divide fMRI into fMRI_fold parts, which is better for calculation .

        '''
    
        total_fmri, _ = self.fmri_loader.load(self.sub)
        FMRI_TR = round(self.config['fmri_tr'] * 1000)
        logging.info(f"Run ridge, sub: {self.sub}, tr_list: {cat_tr_list}, time: {[t * FMRI_TR for t in cat_tr_list]}(ms)")



        logging.info(f"fMRI_shape: {total_fmri.shape}, divide fMRI into {fMRI_fold} parts")
        nTR, nVoxel = total_fmri.shape
        total_fmris = torch.split(total_fmri, round(nVoxel / fMRI_fold), dim=1)

        # cat feature of each story
        feature_all = []

        for ind in self.feature_loader.story_range:
            feature_tmp = self.feature_loader.load_one(ind)
            # because we only have one time_delay, we just cat this one.
            feature_all.append(self.catter.cat(feature_tmp, cat_tr_list, ))

        total_feature = torch.cat(feature_all)
        logging.info(f"cat feature of each story, total_feature: {total_feature.shape}")


        # for each part in fMRI, we train a regression model, and then make them together.
        cls_all = []
        corrs_all = []
        part = 0
        for fmri in tqdm(total_fmris, desc='ridge regression for each part of fMRI...', leave=False):
            
            logging.info(f"{'=' * 80}")
            logging.info(f"Ridge regression starts, fMRI part: {part}")

            # ridge regression
            cls, corrs, _ = self.ridge_cv_model.ridge(X=total_feature, y=fmri, )
            
            # save to tmp
            cls_all.append(cls)
            corrs_all.append(np.array(corrs))

            part += 1

        # save to files
        corrs_all = np.concatenate(corrs_all, axis=1)
        average_corrs_all = corrs_all.mean(0)

        if self.is_corrs_saved:
            savefile = get_corrs_path(self.sub, self.result_path, cat_tr_list)
            savemat(savefile, {'test_corrs': corrs_all})
            logging.info(f"corrs saved: {savefile}, shape: {corrs_all.shape}")
                
            savefile = get_avg_corrs_path(self.sub, self.result_path, cat_tr_list)
            savemat(savefile, {'test_corrs': average_corrs_all})
            logging.info(f"avg_corrs saved: {savefile}, shape: {average_corrs_all.shape}")

        if self.is_weights_saved:
            savefile = get_cls_path(self.sub, self.result_path, cat_tr_list)
            savecls(savefile, cls_all)
            logging.info(f"cls saved: {savefile}")
           
        if self.plot_brain:
            r = self.plot_brain_corrs(average_corrs_all, cat_tr_list)
            if r:
                logging.info(f"brain plot saved.")
            else:
                logging.error(f"brain plot failed to be saved.")
        return


class EncodingModelMetric():
    '''
        This model is used as the base model for different configs, because we need to change our similarity-metric method 
        or else to give a better prediction between our fMRI and word-features.

        We assume that BOLD signals are affected by Gaussian additive noise according to 'Tang 2022'.
    '''
    def __init__(self, 
        sub,
        result_path,

        ) -> None:



        ridge_path = get_cls_path()
        self.ridge_model = loadcls(ridge_path)
        pass

    def score():
        '''
            score pred_y and truth_y, return to a score.
        '''
