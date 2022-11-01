from ctypes.wintypes import WORD
import re
import scipy.io as sio
import numpy as np


word_meg_story = 'fmri_meg_decode_combine/data/word_meg_story/word_meg_story22.pkl'
word_file = 'wordlevel_timealign/derivatives:annotations:time_align:word-level:story_22_word_time.mat'
emb_file = 'gpt_word_embedding/derivatives:annotations:embeddings:gpt:word-level:story_22_word_gpt_0-24_1024.mat'
word_data = sio.loadmat(word_file)
emb_data = sio.loadmat(emb_file)
emb_data = emb_data['data'][18]
# print(emb_data.shape)
word = word_data['word']
print(word.shape)
word_index = 0
a = 0

word_result = word

for i in range(len(word)):
    res = re.findall('[\u4e00-\u9fa5]', word[i])
    if len(res) == 0:
        emb_data = np.delete(emb_data,word_index,axis=0)
        word_result = np.delete(word_result,word_index,axis=0)
        word_index -= 1
        a += 1
    word_index += 1
# print(emb_data)
# print(emb_data.shape)
print(word_result)
print(word_result.size)
print(a)
# print(len(res))
