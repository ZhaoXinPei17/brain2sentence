'''
This file is for beam search decoder. Not only for brain.
used for beam_searching from a set of candidates, and get the most possible sentences from them.

'''
from re import T
import torch
from smn4_album import Album
from smn4_loader import FmriLoader
import numpy as np
from scipy.io import loadmat
from os.path import join
import jieba
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForCausalLM






class NucleusSamplingGenerator(Album):
    '''
        a royal inplementation of Huth 2022.
        Use fMRI of story_i and output the most likely results of text_decoding
    '''
    def __init__(self, sub, **kwargs):
        super().__init__(**kwargs)
        self.sub = sub

        encoding_weights_path = join(self.result_path, f"weight_sub{sub}.mat")
        self.encoding_weights = loadmat(encoding_weights_path)["weights"]

        self.fmri_loader = FmriLoader(**kwargs)

        # initial gpt
        cgpt_model = join(self.gpt_model_path, self.gpt_type)
        self.tokenizer = AutoTokenizer.from_pretrained(cgpt_model)
        self.gpt_model = AutoModelForCausalLM.from_pretrained(cgpt_model, output_hidden_states=True)
        self.spm_model = spm.SentencePieceProcessor(model_file=join(cgpt_model, 'chinese_vocab.model'))

    def forward(self, story_index, word_rates: np.array, **kwargs):
        '''
            sub: subject
            word_rate: fmri_length
        '''
        fmri = self.fmri_loader.load_one(sub=self.sub, story=story_index) # TR * voxels
        
        tr_len = self.ref_length[story_index - 1]
        encoding_model = lambda mat: np.matmul(mat, self.encoding_weights)
        
        seqs = [''] # list of seq
        ranks = [20] # list of int, ranks[s] is rank of seqs[s]
        prompt_length = 20

        cat_words = self.cat_words

        log = self.log
        for t in range(tr_len):
            log.info(f"TR = {t}, seqs = {seqs}")
            for w in range(word_rates[t]):
                # generate
                fathers = []
                continuations = []
                for s in range(len(seqs)):
                    prompt = seqs[s]
                    if prompt == '':
                        continuation_tmps = self.text_generate(prompt, prompt_length, top_k=30) # list
                    else:
                        continuation_tmps = self.text_generate(prompt, prompt_length, top_k=self.gpt_args["top_k"]) # list
                    for c in continuation_tmps:
                        if cat_words(c) != '':
                            continuations.append(c)
                            fathers.append(s)

                    # continuations: c * str; fathers: c * 1
                log.info(continuations); 
                # score each continuation by the likelihood P(R_test | C)
                # 1. embeddings
                log.info("get embeddings...")
                n_cont = len(continuations)
                features = np.empty((n_cont, 1024))
                for c in range(n_cont):
                    prompt = seqs[fathers[c]]
                    # print(continuations[c], prompt); input()
                    features[c] = self.get_feature(continuations[c], prompt, prompt_length)
                
                # 2. scores
                # encoding_model(features): n * V, fmri[t]: V * 1
                log.info("fmri_similarity...")
                cont_scores = self.fmri_similarity(encoding_model(features), fmri[t]) # c * 1

                # choose top_k
                log.info("choose top_k...")
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
                
                # log.info(cont_choose)
                # generate new seqs
                cont_index = (np.where(cont_choose == 1))[0]
                # log.info(cont_index)
                new_seqs = [] # seq_num * 1
                new_scores = []
                for ind in cont_index:
                    # log.info(seqs[fathers[ind]], continuations[ind])
                    seq_tmp = cat_words(seqs[fathers[ind]], continuations[ind])
                    new_seqs.append(seq_tmp)
                    new_scores.append(cont_scores[ind])
            
                # get rank of scores
                ranks = self.scores2rank(np.array(new_scores))
                seqs = new_seqs

    def scores2rank(self, scores):
        '''
            scores: len * 1
            return: ranks: len * 1
        '''
        rank_n = self.max_continuation
        length = scores.shape[0]
        ranks = np.zeros(length)
        indexes = np.argsort(scores)

        for n in range(rank_n):
            low = round(length * n / rank_n)
            high = round(length * (n + 1) / rank_n)
            # print(high, low)
            ranks[indexes[low: high]] += n

        return ranks

    def fmri_similarity(self, nV, fMRI):
        '''
            Return similarity scores of nV and fMRI.
            nV: number * Voxels
            fMRI: voxels
        '''
        fMRI = np.squeeze(fMRI)
        num, v = nV.shape
        assert v == fMRI.shape[0]
        zs = lambda v: (v-v.mean(0))/v.std(0)

        scores = np.empty(num)
        for n in range(num):
            scores[n] = np.dot(zs(nV[n, :]), (zs(fMRI)).T)
            # print(scores[n])
        return scores

    def text_generate(self, prompt, prompt_length, top_k=100, top_p=1, **kwargs):
        '''
            texts_generate: list of (text)
            Params:
                prompt: sequence of prompt_ids
                prompt_length: a limit to prompt. Only last prompt_length words will be feed into GPT
                continuation_length: number of words that will be generated by GPT. This is decided by word_rate model.
        '''
        
        if len(prompt) > prompt_length:
            prompt = prompt[-prompt_length: ]

        if len(prompt) == 0:
            input_ids = self.tokenizer.convert_tokens_to_ids(['<s>'])
        else:
            tokenized_sent = ' '.join(jieba.cut(prompt))
            pieced_sent = self.spm_model.Encode(tokenized_sent, out_type=str)
            input_ids = self.tokenizer.convert_tokens_to_ids(pieced_sent)
            # print(input_ids)

        input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0)

        outputs = self.gpt_model(input_ids,labels=input_ids, ).logits
        mask_pos = outputs.topk(k=top_k, dim=-1, largest=True, sorted=True)[1][0, -1, :]
        
        out_toks = self.tokenizer.convert_ids_to_tokens(mask_pos)
        # print(prompt, out_toks)
        return out_toks

    def get_feature(self, c, seq, prompt_length):
        '''
            get feature of c after seq
        '''
        text = self.cat_words(seq, c)
        model = self.gpt_model

        tokenized_sent = ' '.join(jieba.cut(text))
        pieced_sent = self.spm_model.Encode(tokenized_sent, out_type=str)
        if len(pieced_sent) > prompt_length:
            pieced_sent = pieced_sent[-prompt_length: ]
        input_ids = self.tokenizer.convert_tokens_to_ids(pieced_sent)
        input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0)
        outputs = model(input_ids,labels=input_ids, )
        hiddens = outputs.hidden_states[self.layer_num].detach().numpy()
        # print(hiddens.shape)
        return hiddens[0][-1]

    @staticmethod
    def cat_words(*args) -> str:
        full_t = []
        for t in args:
            if type(t) == str:
                full_t.append(t)
            else: # list
                full_t = full_t + t
        
        return ''.join(full_t).replace('▁', '')



if __name__ == "__main__":
    nucleus_sampling_searcher = NucleusSamplingGenerator(sub="01")
    nucleus_sampling_searcher(story_index=1, word_rates=np.ones(1000, dtype=int),)
