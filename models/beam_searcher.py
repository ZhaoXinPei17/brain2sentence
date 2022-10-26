'''
This file is for beam search decoder. Not only for brain.
used for beam_searching from a set of candidates, and get the most possible sentences from them.

'''
from smn4_album import Album
import numpy as np


class NucleusSamplingGenerator(Album):
    '''
        a royal inplementation of Huth 2022
    '''
    def __init__(self, ):
        super().__init__()
   
    def generate(self, prompt, ):
        pass

    def beam_search(self, scores, ) -> np.array:
        '''
            Search the best top_p from scores.
        '''
        sum_p = 0
        probs = self.score2prob(scores)
        sort_index = (np.argsort(probs))[::-1]

        ind = 0
        while sum_p <= self.top_p:
            sum_p += probs(sort_index[ind])
            ind += 1

        return sort_index[: ind]

    def score2prob(self, scores, ):
        pass


    def run(self, candidates: list, remote_data, k, p, **kwargs) -> np.ndarray:
        '''
            return to index of beam_search
            candidates: candidates of the generate texts
            remote data: outside data using for extra judgement, like fMRI
            k: limit_num of candidates
            p: sum p of nucleus search
        '''
        c_len = len(candidates)
        # get scores of candidates
        scores = self.score_map(candidates, remote_data) # 1*L

        # nucleus search, get the top p percent of the probability mass
        # we treat scores as probabilities
        
        indexs = np.argsort(scores) # 1*L

        return_indexs = []
        sum_tmp = 0

        for i in range(c_len):
            index = indexs[i]
            return_indexs.append(index) 
            
            sum_tmp += scores[index] # add the maximum from scores
            if sum_tmp > p:
                break

        return np.array(return_indexs)

    def score_map(self, candidates, remote_data) -> np.ndarray:
        '''
            calculate scores of all the candidates
        '''
        scores = np.zeros(len(candidates))

        # tmp
        score_std = self.score_std
        for i in range(len(candidates)):
            c = candidates[i]
            scores[i] = score_std(c, remote_data)

        return scores



    


if __name__ == "__main__":
    # test of NucleusSamplingSearcher
    # nucleus_sampling_searcher = NucleusSamplingSearcher(np.random.random())
    pass