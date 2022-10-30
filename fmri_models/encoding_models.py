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

class EncodingModel(Album):
    def __init__(self, sub):
        super().__init__()
        self.sub = sub

    def load_fmri(self):
        fmri_loader = FmriLoader()
        fmri_loader.voxel_top_num = None # encoding models don't need voxel_top
        return fmri_loader(sub=self.sub)

    def load_feature(self):
        feature_loader = FeatureLoader()
        return feature_loader()

    def load_multi_feature(self):
        pass

    def load_test(self):
        pass

class RidgeModel(Album):
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
                valid_fmri, valid_feature, alphas, singcutoff=1e-10)

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


class encoding_base:
    """ 
        perform voxel-wise encoding 
    """

    def __init__(self, args):
        self.method = args['model']['method']
        self.block_shuffle = args['model']['block_shuffle']
        self.blocklen = args['model']['blocklen']
        # self.alphas = args['model']['alphas']
        self.nfold = args['model']['nfold'] # non-cross if nfold == 1
        self.inner_fold = args['model']['inner_fold'] # must > 1, or it would be non-nested cv

        self.fmri_path = args['data']['fmri_path']
        self.fmri_voxel_num = args['data']['fmri_voxel_num']
        self.feature_path = args['data']['feature_path']
        self.feature_type = args['data']['feature_type']
        self.feature_abandon = args['data']['feature_abandon']
        self.n_story = args['data']['n_story']
        self.story_range = range(1, self.n_story + 1)
        self.language = args['data']['language']
        self.test_id = args['data']['test_id'] # no designated test stimuli if test_id == -1

        self.cuda0 = args['exp']['cuda0']
        self.cuda1 = args['exp']['cuda1']
        self.use_cuda = args['exp']['use_cuda']

        self.plot_brain = args['report']['plot_brain']
        self.brain_template = args['report']['brain_template']
        self.result_dir = args['report']['result_dir']

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def load_fmri(self):
        '''
        Return:
            train_fmri: (n_TRs, n_voxels); 
            starts: list, the start index of each story in the fmri_path;
        '''
        train_fmri = torch.tensor([])
        starts = [0]
        for i in tqdm(self.story_range, desc=f'loading fmri from {self.fmri_path} ...'):
            if i == self.test_id:
                continue
            fmri_file = join(self.fmri_path, self.sub, f'story_{i}.mat')
            data = h5py.File(fmri_file, 'r')

            single_fmri = np.array(data['fmri_response'])
            train_fmri = torch.cat([train_fmri, torch.FloatTensor(single_fmri)])
            starts.append(train_fmri.shape[0])
            logging.info(f"story = {i}, single_fmri_tr = {single_fmri.shape[0]}, train_fmri_tr = {train_fmri.shape[0]}")

            
        return train_fmri, starts

    def load_feature(self):
        '''
        Return:
            train_feature: (n_TRs, feature_dim)
        '''
        train_feature = torch.tensor([])
        feature_type = self.feature_type
        feature_abandon = self.feature_abandon
        starts = [0]
        if self.language == 'zh' or self.language == 'en':
            for i in tqdm(self.story_range, desc=f'loading stimulus from {self.feature_path} ...'):
                if i == self.test_id:
                    continue
                feature_file = join(self.feature_path, f'story_{i}.mat')
                
                data = h5py.File(feature_file, 'r')
                single_feature = torch.tensor(np.array(data[feature_type])[: , feature_abandon: ]).transpose(0, 1)
                train_feature = torch.cat([train_feature, single_feature])
                starts.append(train_feature.shape[0])
                
                logging.info(f"story = {i}, single_feature_tr = {single_feature.shape[0]}, train_feature_tr = {train_feature.shape[0]}")

        else:
            raise('Unknown language!')
            
        return train_feature, starts

    def load_multi_feature(self, features):
        '''
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

    def ridge_multidim(self, train_fmri, train_feature, valid_fmri, valid_feature, alphas,
                cuda0=0, cuda1=1, use_cuda=False, singcutoff=1e-10):
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

        if use_cuda:
            alphas = torch.tensor(alphas).cuda(cuda0)
        else:
            alphas = torch.tensor(alphas)

        if use_cuda:
            UR = torch.matmul(U.transpose(0, 1).cuda(cuda1), train_fmri).cuda(cuda0)
            PVh = torch.matmul(valid_feature, V)
        else:
            UR = torch.matmul(U.transpose(0, 1), train_fmri)
            PVh = torch.matmul(valid_feature, V)

        zvalid_fmri = zs(valid_fmri)
        Rcorrs = [] ## Holds training correlations for each alpha
        for a in alphas:
            D = S/(S**2+a**2) ## Reweight singular vectors by the ridge parameter
            if use_cuda:
                pred = torch.matmul(mult_diag(D, PVh, left=False), UR)
            else:
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
        nii_name = self.result_dir+self.language+'_'+self.sub+'_'+self.feature_type+'_'+self.method+'.dscalar.nii'
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

class encoding_cv(encoding_base):
    def __init__(self, args, alphas, sub):
        super(encoding_cv, self).__init__(args)
        self.sub = sub
        self.alphas = alphas
        try:
            self.features = args['data']['features']
        except KeyError: # KeyError: 'features'
            self.features = []
            
    def run_ridge(self):
 
        if len(self.features) > 1:
            total_feature, starts = self.load_multi_feature(self.features)
        else:
            total_feature, starts = self.load_feature()
        total_fmri, starts = self.load_fmri()
        
        if self.method == 'nested_cv':
            corrs = self.ridge_nested_cv(total_fmri, total_feature)
        elif self.method == 'cv':
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
                    Rcorrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, \
                        self.alphas, self.cuda0, self.cuda1, self.use_cuda)
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
        savefile = join(self.result_dir, f"{self.language}_sub{self.sub}_{self.feature_type}_{self.method}.mat")
        scio.savemat(savefile, {'test_corrs':np.array(test_corrs.cpu())})
        savefile = join(self.result_dir, f"{self.language}_sub{self.sub}_{self.feature_type}_{self.method}_average.mat")
        scio.savemat(savefile, {'test_corrs':np.array(test_corrs.mean(0).cpu())})
        return np.array(test_corrs.mean(0).cpu())
        
    def ridge_cv(self, total_fmri, total_feature, starts):
        """ 
        normal cross-validation, which is applicable to situations with
        designated test set
        Return: corrs on test set
        """
        nTR, n_dim = total_feature.shape
        if self.block_shuffle:
            inds = self.block_shuffle_inds(nTR)
        else:
            inds = list(range(nTR))
        
        if self.test_id > -1:
            test_fmri, test_feature = self.load_test()
        else:
            test_len = nTR*self.test_ratio
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
                corrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, \
                                    self.alphas, self.cuda0, self.cuda1, self.use_cuda)
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
                    Rcorrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, \
                        self.alphas, self.cuda0, self.cuda1, self.use_cuda)
                val_corrs.append(torch.stack(Rcorrs))
            val_corrs = torch.stack(val_corrs)  
            max_ind = torch.argmax(val_corrs.mean(2).mean(1))
            bestalpha = self.alphas[max_ind]  
            U,S,V = torch.svd(total_feature)
            # print(U.shape, S.shape, V.shape)
            # input()
            UR = torch.matmul(U.transpose(0, 1), total_fmri)
            wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+bestalpha**2)), UR])
            pred = torch.matmul(test_feature, wt)
            corrs = (zs(pred)*zs(test_fmri)).mean(0)
        
        savefile = join(self.result_dir, f"{self.language}_sub{self.sub}_{self.feature_type}_{self.method}.mat")
        scio.savemat(savefile, {'test_corrs':np.array(corrs.cpu())})
        return np.array(corrs.cpu())


if __name__ == "__main__":

    argp = ArgumentParser()
    argp.add_argument('--exp_config', help='experimental config')
    argp.add_argument('--subs', nargs='+', default=['01'], help='subject label')
    
    cli_args = argp.parse_args()
    args = yaml.load(open(cli_args.exp_config), Loader=yaml.FullLoader)

    alphas = np.logspace(-3, 3, 20)

    for sub in cli_args.subs:
        encoding = encoding_cv(args, alphas, sub)
        encoding.run_ridge()