'''
This file is for beam search decoder. Not only for brain.
used for beam_searching from a set of candidates, and get the most possible sentences from them.

'''
from smn4_album import Album
from smn4_loader import FmriLoader
import numpy as np

class NucleusSamplingGenerator(Album):
    '''
        a royal inplementation of Huth 2022.
        Use fMRI of story_i and output the most likely results of text_decoding
    '''
    def __init__(self, ):
        super().__init__()

    def forward(self, sub, story_index, word_rates: np.array):
        '''
            sub: subject
            word_rate: fmri_length
        '''
        fmri_loader = FmriLoader()
        fmri = fmri_loader.load_one(sub=sub, story=story_index) # TR * voxels
        
        tr_len = self.ref_length[story_index - 1]
        encoding_model = 0


        seqs = [''] # list of seq
        ranks = [20] # list of int, ranks[s] is rank of seqs[s]

        for t in range(tr_len):
            fathers = []
            # generate
            continuations = []
            for s in range(len[seqs]):
                prompt = seqs[s]
                continuation_tmps = self.text_generate(prompt, prompt_length, continuation_length) # list
                for c in continuation_tmps:
                    continuations.append(c)
                    fathers.append(s)
            # continuations: c * str; fathers: c * 1

            # score each continuation by the likelihood P(R_test | C)
            # 1. embeddings
            features = []
            for c in continuations:
                features.append(self.get_features(c))
            features = np.array(features) # ??????????????
            # 2. scores
            cont_scores = self.similarity(encoding_model(features), fmri[t]) # c * 1

            # choose top_k
            top_k = self.continuation_top_k
            cont_choose = np.zeros(cont_scores.shape[0])
            for i in (np.argsort(cont_scores))[::-1]:
                if ranks[fathers[i]] > 0 and top_k > 0:
                    cont_choose[i] = 1
                    ranks[fathers[i]] -= 1
                    top_k -= 1
                elif top_k == 0:
                    break
                else: # rank[fathers[i]] == 0
                    continue

            # generate new seqs
            cont_index, _ = np.where(cont_choose == 1)
            new_seqs = [] # seq_num * 1
            new_scores = []
            for i in cont_index:
                seq_tmp = seqs[fathers[i]] + continuations[i]
                new_seqs.append(seq_tmp)
                new_scores.append(cont_scores[i])
            
            # get rank of scores
            ranks = self.scores2rank(np.array(new_scores))
            seqs = new_seqs

    def scores2rank(self, scores):
        '''
            scores: len * 1
            return: ranks: len * 1
        '''
        rank_n = self.max_continuation
        length = scores.shapes[0]
        ranks = np.zeros(length)
        indexes = np.argsort(scores)

        for n in range(rank_n):
            low = np.round(length * n / rank_n)
            high = np.round(length * (n + 1) / rank_n)
            ranks[indexes[low: high]] += n

        return ranks


    def similarity(self, A, B):
        '''
            A and B: x * y np.array
        '''
        assert A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1]


    def text_generate(self, ):
        '''
            texts_generate: list of (text, score)
        '''
        if len(texts_scores) == 0:
            # free generate
            pass
        else:
            pass
  


if __name__ == "__main__":
    # test of NucleusSamplingSearcher
    # nucleus_sampling_searcher = NucleusSamplingSearcher(np.random.random())
    pass