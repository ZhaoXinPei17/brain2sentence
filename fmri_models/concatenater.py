
import torch
import numpy as np
from typing import List

class Concatenater:
    '''
        For features or fMRI, we need to concatenate different vectors of different time.
    '''

    def cat(self, vectors: torch.Tensor, t_sequence: List[int]):
        '''
            concatenate vectors by t_sequence
            Arg:
                vectors (`Tensor((nTR, dim))`):
                    data being concatenated
                t_sequence (`List[int]`): 
                    used for concatenating.
                    for example: [1,2,4]: t+1, t+2, t+4
            Return:
                cat_vectors (`Tensor((nTR, n * dim))`), 
                where n = len(t_sequence)
        '''

        nTR, dim = vectors.shape
        nT = len(t_sequence)
        
        # if `a` = [[1,2], [2,3]], we cannot concatenate when tr = 1, so we need to make `a` = [[1,2], [2,3], [0,0]]
        # get the max abs of t_sequence, and add zeros to the head and tail of vectors.
        tr_attd = max([abs(t) for t in t_sequence])
        zeros = torch.zeros((tr_attd, dim))
        vectors = torch.cat((zeros, vectors, zeros), axis=0, ) # (tr_attd + nTR + tr_attd, dim)

        vectors_cat = torch.zeros((nTR, dim * nT))
        for t in range(nT):     
            vectors_cat[:, dim * t: dim * (t + 1)] = \
                vectors[tr_attd + t_sequence[t]: tr_attd + t_sequence[t] + nTR, :]

        return vectors_cat
