'''
    Ridge Regression models for every task in our work.

    RidgeRegression:
        base ridge regression model

    RidgeModel:
        a simple ridge regression model 


    RidgeCVModel:
        a widely used ridge regression model with cross validation. 
        "cv" for chosen-test-datasets( or generate test-dataset for only once)
        "nested_cv" divide datasets into folds, and each fold has a chance to be tested by a model trained by the others
'''
from collections import Iterable
import torch
from torch import Tensor
import numpy as np
import itertools as itools
from functools import reduce
from tqdm import tqdm
import random
from typing import List

import logging

from .smn4_utils import *

class RidgeRegression():
    def __init__(self, alpha, ) -> None:
        self.alpha = alpha
        self.params = {}

    def fit(self, X, y):
        self.X_dim = get_dim(X.shape)
        self.y_dim = get_dim(y.shape)
        if self.X_dim == 1:
            return self.fit_1dim(X, y)
        else:
            return self.fit_multidim(X, y)
    
    def fit_1dim(self, X: Tensor, y: Tensor, ):
        S = torch.matmul(X.transpose(0,1), X)
        S = X/(S + self.alpha**2)
        W = torch.matmul(S.transpose(0,1), y)
        self.params = {
                "W": W
            }

    def fit_multidim(self, X: Tensor, y: Tensor, singcutoff=1e-10):
        """
            this function can be used on features with more than 1 dimension, 
            such as word embeddings (BERT, elmo, etc.), pos tags, 
            or other semantic features
        """
        # print(X.dtype, y.dtype); input()
        U,S,V = torch.svd(X)

        ngoodS = torch.sum(S>singcutoff)
        U = U[:,:ngoodS]
        S = S[:ngoodS]
        V = V[:,:ngoodS]

        UR = torch.matmul(U.transpose(0, 1), y)
        # input()
        self.params = { 
            "S": S, 
            "V": V, 
            "UR": UR, 
            }

    def predict(self, X):
        params = self.params
        if self.X_dim == 1:
            return self.predict_1dim(X, **params)
        else:
            return self.predict_multidim(X, **params)

    def predict_1dim(self, X, W, **params):
        pred = torch.matmul(X, W)
        return pred

    def predict_multidim(self, X: Tensor, S, V, UR, **params) -> Tensor:      
        PVh = torch.matmul(X, V)
        D = S/(S**2 + self.alpha**2) # Reweight singular vectors by the ridge parameter 
        pred = torch.matmul(mult_diag(D, PVh, left=False), UR)              
        return pred

    def score(self, pred_y: Tensor, truth_y: Tensor) -> Tensor:
        '''
            Return:
                Rcorr: shape of (1, y_dim)
        '''
        zs = lambda v: (v-v.mean(0))/v.std(0)
        zs_pred_y = zs(pred_y)
        zs_truth_y = zs(truth_y)
        Rcorr = (zs_pred_y * zs_truth_y).mean(0)            
        Rcorr[torch.isnan(Rcorr)] = 0
        return Rcorr

class RidgeModel():
    '''
        Commom Ridge model.
        This ridge model will use itself as the test dataset.
    '''
    def __init__(self, alphas, **kwargs):
        self.alphas = alphas
    
    def ridge(self, X, y):
        # initial
        # y = y if type(y) is Tensor else torch.tensor(y)
        # X = X if type(X) is Tensor else torch.tensor(X)
        
        X_dim = get_dim(X.shape)
        y_dim = get_dim(y.shape)

        cls_all = []
        Rcorrs = []
        for a in tqdm(self.alphas, desc=f'doing nested ridge regression, X_shape: {X.shape}, y_shape: {y.shape}', leave=False):
            cls = RidgeRegression(a)
            
            cls.fit(X, y)
            py = cls.predict(X)
            corrs = cls.score(py, y)
            
            cls_all.append(cls)
            Rcorrs.append(corrs)

        Rcorrs = torch.stack(Rcorrs) # shape: (alpha, y_dim)

        avg_Rcorrs = Rcorrs if y_dim == 1 else Rcorrs.mean(1)
        
        max_ind = torch.argmax(avg_Rcorrs)
        best_cls = cls_all[max_ind]
        best_avg_Rcorrs = avg_Rcorrs[max_ind]

        return best_cls, best_avg_Rcorrs

class RidgeCVModel():
    '''
        Commom RidgeCV model.
    '''
    def __init__(self, 
        alphas=None, 
        block_shuffle=None, 
        encoding_method=None, 
        nfold=None, 
        inner_fold=None, 
        train_ratio=None,
        test_ratio=None,
        test_y=None,
        test_X=None,
        **ridge_cv_config):

        self.alphas = alphas
        self.block_shuffle = block_shuffle
        self.encoding_method = encoding_method
        self.nfold = nfold
        self.inner_fold = inner_fold
        self.test_y = test_y
        self.test_X = test_X
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def ridge(self, X, y):
        '''
            Ridge of X and y, estimating `X \\theta = y`

            Arg:
                X: Tensor
                y: Tensor

            Return: 
                weight: `\\theta`
                corrs: predict_scores of each element in y, showing the correlation of elements in X and y
                avg_corrs: average predict_scores of each element in y
        '''
        # y = y if type(y) is Tensor else torch.tensor(y)
        #  X = X if type(X) is Tensor else torch.tensor(X)

        if self.encoding_method == 'nested_cv':
            cls, corrs, avg_corrs = self.ridge_nested_cv(y, X)
        elif self.encoding_method == 'cv':
            cls, corrs, avg_corrs = self.ridge_cv(y, X)
        else:
            raise('Unsupported training method (only "nested_cv" and "cv" are supported)!')
    
        return cls, corrs, avg_corrs

    def ridge_nested_cv(self, y: Tensor, X: Tensor):
        """ 
            nested cross-validation, which is applicable to situations without
            designated test set
            
            Return: 
                `cls`: ridge regression model using the best `\\alpha`
                `corrs`: correlation score of best `cls`
                `corrs_average`: average corrs of `corrs`
        """
        assert X.shape[0] == y.shape[0]
        
        nTR = X.shape[0]
        X_dim = 1 if X.ndim == 1 else X.shape[1]
        y_dim = 1 if y.ndim == 1 else y.shape[1]

        inds = self.block_shuffle_inds(nTR) if self.block_shuffle else list(range(nTR))
        foldlen = round(nTR / self.nfold)

        test_corrs = []
        best_cls = None
        best_corr = 0

        logging.info(f"Nested ridge regression, X_shape: {X.shape}, y_shape: {y.shape}")
        for fold in tqdm(range(self.nfold), desc=f'doing nested ridge regression, X_shape: {X.shape}, y_shape: {y.shape}', leave=False, ):

            test_inds = inds[fold * foldlen: (fold +1 ) * foldlen]
            inner_inds = inds[0: fold * foldlen] + inds[(fold + 1) * foldlen:]
            infoldlen = round(len(inner_inds)/self.inner_fold)
            
            test_y = y[test_inds]
            test_X = X[test_inds]

            val_corrs = []
            # cls_alls = []
            for infold in tqdm(range(self.inner_fold), desc=f"doing cross-validation for each infold...", leave=False, ):

                val_inds = inner_inds[infold*infoldlen:(infold+1)*infoldlen]
                train_inds = inner_inds[0:infold*infoldlen] + inner_inds[(infold+1)*infoldlen:]
                
                train_y, valid_y = y[train_inds], y[val_inds]
                train_X, valid_X = X[train_inds], X[val_inds]
                
                _, Rcorrs = self.ridge_run(train_y, train_X, valid_y, valid_X, self.alphas)
                
                val_corrs.append(torch.stack(Rcorrs))

            val_corrs = torch.stack(val_corrs) # shape: (inner_fold, alpha_num, y.shape[1])
            
            # if y_dim == 1, we do not need to calculate the average val_corrs of dimension 2
            max_ind = torch.argmax(val_corrs.mean(0)) if y_dim == 1 \
                else torch.argmax(val_corrs.mean(2).mean(0)) 

            bestalpha = self.alphas[max_ind]  
            # test
            cls, corrs = self.ridge_run(y[inner_inds], X[inner_inds], test_y, test_X, bestalpha, is_test=True)

            # if y.shape[1] != 1, we just see corrs > 0
            corr_sum = corrs if y_dim == 1 \
                else sum(corrs[(np.where(corrs > 0))[0]])
            
            logging.info(f"fold = {fold}, corr_sum = {corr_sum}, while best_corr = {best_corr}")
            if corr_sum > best_corr:
                best_cls = cls
                best_corr = corr_sum

            test_corrs.append(corrs) # len: nfold

        test_corrs = torch.stack(test_corrs) # dim: (nfold, y_dim)

        corrs = test_corrs.numpy()
        corrs_average = corrs.mean(0)
        return best_cls, corrs, corrs_average
        
    def ridge_cv(self, y, X):
        """ 
            normal cross-validation, which is applicable to situations with
            designated test set
            Return: corrs on test set
        """
        assert y.shape[0] == X.shape[0]
        assert self.train_ratio is not None
        nTR, = X.shape[0],
        
        inds = self.block_shuffle_inds(nTR) if self.block_shuffle else list(range(nTR))
        
        # get test_datasets
        if self.test_id > -1:
            test_y, test_X = self.test_y, self.test_X ##########
        else:
            test_len = round(nTR * self.test_ratio)
            test_inds = inds[: test_len]
            test_y = y[test_inds]
            test_X = X[test_inds]
            inds = inds[test_len:]


        if self.nfold == 1:
            train_inds = inds[:len(inds)*self.train_ratio]
            valid_inds = inds[len(inds)*self.train_ratio:]
            train_y = y[train_inds]
            train_X = X[train_inds]
            valid_y = y[valid_inds]
            valid_X = X[valid_inds]
            
            _, corrs = self.ridge_run(train_y, train_X, valid_y, valid_X, self.alphas)

            max_ind = torch.argmax(corrs.mean(0))
            bestalpha = self.alphas[max_ind]  
            
            cls, corrs = self.ridge_run(y, X, test_y, test_X, bestalpha, is_test=True)
        
        else:
            val_corrs = []
            foldlen = round(len(inds)/self.nfold)
            
            for fold in range(self.nfold):
                val_inds = inds[fold*foldlen:(fold+1)*foldlen]
                train_inds = inds[0:fold*foldlen]+inds[(fold+1)*foldlen:]
                train_y, valid_y = y[train_inds], y[val_inds]
                train_X, valid_X = X[train_inds], X[val_inds]
                
                _, Rcorrs = self.ridge_run(train_y, train_X, valid_y, valid_X, self.alphas)
                
                val_corrs.append(torch.stack(Rcorrs))
            
            val_corrs = torch.stack(val_corrs) # shape: (inner_fold, alpha_num, y_dim)
            max_ind = torch.argmax(val_corrs.mean(2).mean(0))
            bestalpha = self.alphas[max_ind]  
            
            cls, corrs = self.ridge_run(y, X, test_y, test_X, bestalpha, is_test=True)
        
        corrs_average = corrs.mean(0)
        return cls, corrs, corrs_average

    def ridge_run(self, 
        train_y: Tensor, 
        train_X: Tensor, 
        valid_y: Tensor, 
        valid_X: Tensor, 
        alphas, 
        is_test=False):
        
        """
            this function can be used on features of all dimensions, such as word embeddings (BERT, elmo, etc.), 
            pos tags, or other semantic features

            Return:
                cls_all: list of ridge_regression_models for each alpha in alphas, len = len(alphas)
                Rcorrs: list of Rcorrs for each alpha in alphas. len = len(alphas), Rcorr.shape = (y.shape[1])
        """
        
        # initial
        if train_X.dim() == 1:
            train_X = train_X.reshape(-1, 1)
            valid_X = valid_X.reshape(-1, 1)
        if train_y.dim() == 1:
            train_y = train_y.reshape(-1, 1)
            valid_y = valid_y.reshape(-1, 1)

        assert train_X.shape[0] == train_y.shape[0], f"train_X.shape: {train_X.shape}, train_y.shape: {train_y.shape}"

        cls_all = []
        Rcorrs = [] # hold train correlations of alphas

        alphas = alphas if isinstance(alphas, Iterable) else [alphas] # if alphas is not iterable, change it

        tqbar = tqdm(alphas, desc=f"training ridge regression model for each alpha...", leave=False, ) if not is_test else alphas
        
        for a in tqbar:
            cls = RidgeRegression(alpha=a, )
            
            # train regression model
            cls.fit(X=train_X, y=train_y)
            
            # get valid, and score
            pvalid_y = cls.predict(valid_X)
            Rcorr = cls.score(pvalid_y, valid_y)

            cls_all.append(cls)
            Rcorrs.append(Rcorr)
        
        if is_test:
            return cls_all[0], Rcorrs[0]
        else:
            return cls_all, Rcorrs

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

