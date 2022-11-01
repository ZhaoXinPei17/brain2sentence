# import mat4py
# mat4py.loadmat('data_meg_fmri/derivatives_annotations_embeddings_gpt_word-level_story_10_word_gpt_0-24_1024.mat')
import scipy.io as sio
mat_path = 'derivatives_annotations_embeddings_gpt_word-level_story_1_word_gpt_0-24_1024.mat'
#mat_path = 'derivatives_annotations_embeddings_gpt_word-level_story_10_word_gpt_0-24_1024.mat'
mat_path = 'wordlevel_timealign/derivatives:annotations:time_align:word-level:story_1_word_time.mat'
#mat_path = 'story_1_pku_fmri.mat'

mat_data = sio.loadmat(mat_path)

start  = mat_data['start']
end = mat_data['end']
word = mat_data['word']
print(start[0][-10:])
print(end[0][-10:])
print(word.shape)
# x = mat_data['data']
# print(x.shape)

#直接输出mat文件的内容
#print(mat_data)
#输出mat数据的key
print(mat_data.keys())
#print(mat_data.values())

