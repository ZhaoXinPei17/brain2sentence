# init

import scipy.io as scio
import torch
import numpy as np
from argparse import ArgumentParser
import yaml
import itertools as itools
from utils import mult_diag
from functools import reduce
from tqdm import tqdm
import h5py
import random
import nibabel as nib
import os
from os.path import join
from smn4_album import Album
from smn4_loader import FmriLoader, FeatureLoader

zs = lambda v: (v-v.mean(0))/v.std(0)

class RidgeModel(Album):
    '''
        Ridge model.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, train_fmri, train_feature, 
        valid_fmri, valid_feature, alphas, singcutoff=1e-10):
        _, n_dim = train_feature.shape
        if n_dim == 1:
            return self.ridge_1dim(train_fmri, train_feature, 
                valid_fmri, valid_feature, alphas,)
        else:
            return self.ridge_multidim(train_fmri, train_feature, 
                valid_fmri, valid_feature, alphas, singcutoff)

    def ridge_multidim(self, train_fmri, train_feature, valid_fmri, valid_feature, alphas, singcutoff=1e-10):
        """
            this function can be used on features with more than 1 dimension, 
            such as word embeddings (BERT, elmo, etc.), pos tags, 
            or other semantic features
        """
        
        U,S,V = torch.svd(train_feature) 
        
        ngoodS = torch.sum(S>singcutoff)
        U = U[:,:ngoodS]
        S = S[: ngoodS]
        V = V[:,:ngoodS]

        alphas = torch.tensor(alphas)

        UR = torch.matmul(U.transpose(0, 1), train_fmri)
        PVh = torch.matmul(valid_feature, V)

        zvalid_fmri = zs(valid_fmri)
        Rcorrs = [] # Holds training correlations for each alpha
        for a in alphas:
            D = S/(S**2+a**2) # Reweight singular vectors by the ridge parameter
            pred = torch.matmul(mult_diag(D, PVh, left=False), UR)
            Rcorr = (zvalid_fmri*zs(pred)).mean(0)                
            Rcorr[torch.isnan(Rcorr)] = 0
            Rcorrs.append(Rcorr)
        
        return Rcorrs

    def ridge_1dim(self, train_fmri, train_feature, valid_fmri, valid_feature, alphas):
        Rcorrs = []
        for alpha in alphas:
            S = torch.matmul(train_feature.transpose(0,1), train_feature)
            S = train_feature/(S+alpha**2)
            W = torch.matmul(S.transpose(0,1), train_fmri)
            pred = torch.matmul(valid_feature, W)
            zPresp = zs(valid_fmri)
            Rcorr = (zPresp*zs(pred)).mean(0)
            Rcorr[torch.isnan(Rcorr)] = 0
            Rcorrs.append(Rcorr)
        
        return Rcorrs

class EncodingModel(Album):
    """ 
        perform voxel-wise encoding 
    """
    def __init__(self, sub, **kwargs):
        super().__init__(**kwargs)
        self.sub = sub
        self.fmri_loader = FmriLoader(**kwargs)
        self.feature_loader = FeatureLoader(**kwargs)

    def load_fmri(self):
        fmri_loader = self.fmri_loader
        return fmri_loader.load(sub=self.sub)

    def load_feature(self):
        feature_loader = self.feature_loader
        return feature_loader.load()

    def load_multi_feature(self, features):
        '''
        need rewriting.
        Return:
            train_feature: (n_features, n_TRs, feature_dim)
        '''
        
        if self.language == 'zh' or self.language == 'en':
            all_features = []
            for feat in features:
                train_feature = torch.tensor([])
                feature_path = self.feature_path + feat
                starts = [0]
                for i in tqdm(self.story_range, desc=f'loading stimulus from {feature_path} ...'):
                    if i == self.test_id:
                        continue
                    feature_file = join(feature_path, f'story_{i}.mat')
                    data = h5py.File(feature_file, 'r')
                    single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
                    train_feature = torch.cat([train_feature, single_feature])
                    starts.append(train_feature.shape[0])
                all_features.append(train_feature)
            all_features = torch.stack(all_features, 1).reshape(train_feature.shape[0], len(features))
        else:
            raise('Unknown language!')
            
        return all_features, starts

    def load_test(self):
        feature_file = join(self.feature_path, f'story_{self.test_id}.mat')
        data = h5py.File(feature_file, 'r')
        test_feature = torch.tensor(data['word_feature']).transpose(0, 1)
        fmri_file = self.fmri_path+'/story_'+str(self.test_id)+'.mat'
        data = h5py.File(fmri_file, 'r')
        test_fmri = torch.tensor(data['fmri_response'])
        return test_fmri, test_feature

    def ridge_multidim(self, train_fmri, train_feature, valid_fmri, valid_feature, alphas, singcutoff=1e-10):
        """
            this function can be used on features with more than 1 dimension, 
            such as word embeddings (BERT, elmo, etc.), pos tags, 
            or other semantic features
        """
        
        U,S,V = torch.svd(train_feature) #cuda 1
        
        ngoodS = torch.sum(S>singcutoff)
        U = U[:,:ngoodS]
        S = S[:ngoodS]
        V = V[:,:ngoodS]

        alphas = torch.tensor(alphas)

        UR = torch.matmul(U.transpose(0, 1), train_fmri)
        PVh = torch.matmul(valid_feature, V)

        zvalid_fmri = zs(valid_fmri)
        Rcorrs = [] ## Holds training correlations for each alpha
        for a in alphas:
            D = S/(S**2+a**2) ## Reweight singular vectors by the ridge parameter
            
            pred = torch.matmul(mult_diag(D, PVh, left=False), UR)
            Rcorr = (zvalid_fmri*zs(pred)).mean(0)                
            Rcorr[torch.isnan(Rcorr)] = 0
            Rcorrs.append(Rcorr)
        
        return Rcorrs

    def ridge_1dim(self, train_fmri, train_feature, valid_fmri, valid_feature, alphas):
        Rcorrs = []
        for alpha in alphas:
            S = torch.matmul(train_feature.transpose(0,1), train_feature)
            S = train_feature/(S+alpha**2)
            W = torch.matmul(S.transpose(0,1), train_fmri)
            pred = torch.matmul(valid_feature, W)
            zPresp = zs(valid_fmri)
            Rcorr = (zPresp*zs(pred)).mean(0)
            Rcorr[torch.isnan(Rcorr)] = 0
            Rcorrs.append(Rcorr)
        
        return Rcorrs

    def plot_brain_corrs(self, corrs):
        base = nib.load(self.brain_template)
        corr_nii = nib.Cifti2Image(corrs.reshape(1, self.fmri_voxel_num), base.header)
        nii_name = self.result_path+self.language+'_'+self.sub+'_'+self.feature_type+'_'+self.method+'.dscalar.nii'
        corr_nii.to_filename(nii_name)

    def block_shuffle_inds(self, nTR):
        allinds = range(nTR)
        if self.block_shuffle:
            blocklen = 100
            indblocks = list(zip(*[iter(allinds)]*blocklen))
            if nTR%blocklen != 0:
                indblocks.append(range(len(indblocks)*blocklen, nTR))
            random.shuffle(indblocks)
        # return np.array(list(itools.chain(*indblocks)))
        return list(itools.chain(*indblocks))

class EncodingCVModel(EncodingModel):

    def __init__(self, alphas, is_corrs_saved=False, is_weights_saved=False, **kwargs):
        super().__init__(**kwargs)
        self.alphas = alphas
        self.is_corrs_saved = is_corrs_saved
        self.is_weights_saved = is_weights_saved
        try:
            self.features = self.args['data']['features']
        except KeyError: # KeyError: 'features'
            self.features = []
            
    def run_ridge(self):
 
        if len(self.features) > 1:
            total_feature, starts = self.load_multi_feature(self.features)
        else:
            total_feature, starts = self.load_feature()
        total_fmri, starts = self.load_fmri()
        
        if self.encoding_method == 'nested_cv':
            corrs = self.ridge_nested_cv(total_fmri, total_feature)
        elif self.encoding_method == 'cv':
            corrs = self.ridge_cv(total_fmri, total_feature, starts)
        else:
            raise('Unsupported training method (only "nested_cv" and "cv" are supported)!')
        
        if self.plot_brain:
            self.plot_brain_corrs(corrs)

    def ridge_nested_cv(self, total_fmri, total_feature):
        """ 
            nested cross-validation, which is applicable to situations without
            designated test set
            Return: corrs on all out test set
        """
        # logging.info((total_feature.shape, total_fmri.shape))
        nTR, n_dim = total_feature.shape
        if self.block_shuffle:
            inds = self.block_shuffle_inds(nTR)
        else:
            inds = list(range(nTR))
        foldlen = int(nTR / self.nfold)

        test_corrs = []
        for fold in tqdm(range(self.nfold), desc='doing nested ridge regression...'):
            test_inds = inds[fold * foldlen: (fold +1 ) * foldlen]
            inner_inds = inds[0: fold * foldlen] + inds[(fold + 1) * foldlen:]
            infoldlen = int(len(inner_inds)/self.inner_fold)
            
            test_fmri = total_fmri[test_inds]
            test_feature = total_feature[test_inds]

            val_corrs = []
            for infold in range(self.inner_fold):
                val_inds = inner_inds[infold*infoldlen:(infold+1)*infoldlen]
                train_inds = inner_inds[0:infold*infoldlen]+inner_inds[(infold+1)*infoldlen:]
                train_fmri = total_fmri[train_inds]
                valid_fmri = total_fmri[val_inds]
                train_feature = total_feature[train_inds]
                valid_feature = total_feature[val_inds]
                if n_dim == 1:
                    Rcorrs = self.ridge_1dim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas)
                else:
                    Rcorrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas, )
                val_corrs.append(torch.stack(Rcorrs))
            val_corrs = torch.stack(val_corrs)  
            max_ind = torch.argmax(val_corrs.mean(2).mean(0))
            bestalpha = self.alphas[max_ind]  
            U,S,V = torch.svd(total_feature[inner_inds])
            UR = torch.matmul(U.transpose(0, 1), total_fmri[inner_inds])
            wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+bestalpha**2)), UR])
            pred = torch.matmul(test_feature, wt)
            corrs = (zs(pred)*zs(test_fmri)).mean(0)
            test_corrs.append(corrs)

        test_corrs = torch.stack(test_corrs)
        if self.is_corrs_saved:
            savefile = join(self.result_path, f"sub{self.sub}.mat")
            scio.savemat(savefile, {'test_corrs':np.array(test_corrs)})
            savefile = join(self.result_path, f"sub{self.sub}_average.mat")
            scio.savemat(savefile, {'test_corrs':np.array(test_corrs.mean(0))})
        return np.array(test_corrs.mean(0))
        
    def ridge_cv(self, total_fmri, total_feature, starts):
        """ 
            normal cross-validation, which is applicable to situations with
            designated test set
            Return: corrs on test set
        """
        assert total_fmri.shape[0] == total_feature.shape[0]
        nTR, n_dim = total_feature.shape
        if self.block_shuffle:
            inds = self.block_shuffle_inds(nTR)
        else:
            inds = list(range(nTR))
        
        if self.test_id > -1:
            test_fmri, test_feature = self.load_test()
        else:
            test_len = round(nTR*self.test_ratio)
            test_inds = inds[:test_len]
            test_fmri = total_fmri[test_inds]
            test_feature = total_feature[test_inds]
            inds = inds[test_len:]

        if self.nfold == 1:
            train_inds = inds[:len(inds)*self.train_ratio]
            valid_inds = inds[len(inds)*self.train_ratio:]
            train_fmri = total_fmri[train_inds]
            train_feature = total_feature[train_inds]
            valid_fmri = total_fmri[valid_inds]
            valid_feature = total_feature[valid_inds]
            if n_dim == 1:
                corrs = self.ridge_1dim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas)
            else:
                corrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas, )
        else:
            val_corrs = []
            foldlen = int(len(inds)/self.nfold)
            for fold in range(self.nfold):
                val_inds = inds[fold*foldlen:(fold+1)*foldlen]
                train_inds = inds[0:fold*foldlen]+inds[(fold+1)*foldlen:]
                train_fmri = total_fmri[train_inds]
                valid_fmri = total_fmri[val_inds]
                train_feature = total_feature[train_inds]
                valid_feature = total_feature[val_inds]
                if n_dim == 1:
                    Rcorrs = self.ridge_1dim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas)
                else:
                    Rcorrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas, )
                val_corrs.append(torch.stack(Rcorrs))
            
            val_corrs = torch.stack(val_corrs)  
            max_ind = torch.argmax(val_corrs.mean(2).mean(1))
            bestalpha = self.alphas[max_ind]  
            U,S,V = torch.svd(total_feature) #; print(U.shape, S.shape, V.shape); input()
            UR = torch.matmul(U.transpose(0, 1), total_fmri)
            wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+bestalpha**2)), UR])
            pred = torch.matmul(test_feature, wt)
            corrs = (zs(pred)*zs(test_fmri)).mean(0)
        
        if self.is_weights_saved:
            savefile = join(self.result_path, f"weight_sub{self.sub}.mat")
            scio.savemat(savefile, {'weights':np.array(wt)})
        if self.is_corrs_saved:
            savefile = join(self.result_path, f"sub{self.sub}.mat")
            scio.savemat(savefile, {'test_corrs':np.array(corrs)})
        return np.array(corrs)

if __name__ == "__main__":

    alphas = np.logspace(-3, 3, 20)

    for sub in ['01']:
        encoding = EncodingCVModel(n_story=10, encoding_method='cv', sub=sub, alphas=alphas, is_weights_saved=True)
        encoding.run_ridge()