from ctypes.wintypes import WORD
import re
import scipy.io as sio
import numpy as np
import pickle as p
import chardet

# TODO 去除emb中不含在meg中的词是否合理？去除的是词频低的，所以理论上对连续语义解码影响不大
# 去除原始story word embedding中的标点和多余的词（相比于目标meg）
file_num = 60
file_index = 1
word_meg_file = '/home/jczheng/fmri_meg_decode_combine/data/word_meg_story/word_meg_story22.pkl'
word_meg = open(word_meg_file, 'rb')
word_meg = p.load(word_meg)

word_file = '/home/jczheng/fmri_meg_decode_combine/data/wordlevel_timealign/derivatives:annotations:time_align:word-level:story_22_word_time.mat'
word_origin = sio.loadmat(word_file)
word_origin = word_origin['word']

emb_file = '/home/jczheng/fmri_meg_decode_combine/data/gpt_word_embedding/derivatives:annotations:embeddings:gpt:word-level:story_22_word_gpt_0-24_1024.mat'
emb = sio.loadmat(emb_file)
# load 18th gpt
emb = emb['data'][18]

# 去除emb的标点符号
# word_result = word_origin
# word_index = 0
# for i in range(len(word_origin)):
#     res = re.findall('[\u4e00-\u9fa5]', word_origin[i])
#     if len(res) == 0:
#         emb = np.delete(emb, word_index, axis=0)
#         word_result = np.delete(word_result, word_index, axis=0)
#         word_index -= 1
#     word_index += 1
# print(word_result)
# print(word_result.shape)
# print(emb.shape)

# 去除emb中不存在于meg数据中的词
word_origin_index = 0  
emb_index = 0  # word_result和emb从始至终维度相同，所以用同一个索引
word_result = word_origin
for word_meg_index in range(len(word_meg)):
    while True:
        if word_origin[word_origin_index].strip() == word_meg[word_meg_index].strip():
            word_origin_index += 1
            emb_index += 1
            break
        else:
            emb = np.delete(emb, emb_index, axis=0)
            word_result = np.delete(word_result, emb_index, axis=0)
            word_origin_index += 1
# 删除末尾的标点和其它字符
for i in range(emb_index, emb.shape[0]):
    emb = np.delete(emb, emb_index, axis=0)
    word_result = np.delete(word_result, emb_index, axis=0)

print(word_result)
print(word_result.shape)
print(emb.shape)