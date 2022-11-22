
import yaml
import os
from os.path import join
import h5py
import logging

CONFIG = '/sdb/xpzhao/zxps/brain2sentence/fmri_config.yaml'

def load_ref_TRs(file_root):
    res = []
    for i in range(1, 61):
        data = h5py.File(join(file_root, f'story_{i}.mat'), 'r')
        res.append(data['word_feature'].shape[1])
    return res

class Album():
    '''
        This class is used as the root_class of ALL classes in brain2sentence.
    '''
    args = yaml.load(open(CONFIG), Loader=yaml.FullLoader)

    def __init__(self, 
        fmri_path = args['fmri_path'], 
        result_path = args['result_path'], 
        encoding_model_path = args['encoding_model_path'], 
        fmri_feature_corrs_path = args['fmri_feature_corrs_path'], 
        embedding_path = args['embedding_path'], 
        feature_path = args['feature_path'], 
        time_path = args['time_path'], 
        ref_time_path = args['ref_time_path'], 
        encoding_method = args['encoding_method'], 
        block_shuffle = args['block_shuffle'], 
        blocklen = args['blocklen'], 
        nfold = args['nfold'], 
        inner_fold = args['inner_fold'], 
        train_ratio = args['train_ratio'], 
        test_ratio = args['test_ratio'], 
        word_rate_method = args['word_rate_method'], 
        layer_num = args['layer_num'], 
        convolve_type = args['convolve_type'], 
        duration_time_type = args['duration_time_type'], 
        fmri_tr = args['fmri_tr'], 
        fmri_voxel_num = args['fmri_voxel_num'], 
        voxel_top_num = args['voxel_top_num'], 
        feature_type = args['feature_type'], 
        feature_abandon = args['feature_abandon'], 
        n_story = args['n_story'], 
        test_id = args['test_id'], 
        continuation_top_k = args['continuation_top_k'], 
        max_continuation = args['max_continuation'], 
        top_p = args['top_p'], 
        gpt_model_path = args['gpt_model_path'], 
        gpt_type = args['gpt_type'], 
        gpt_args = args['gpt_args'], 
        use_cuda = args['use_cuda'], 
        logging_file = args['logging_file'], 
        plot_brain = args['plot_brain'], 
        brain_template = args['brain_template'], 
        **kwargs):

        Args = dict()

        Args['fmri_path'] = fmri_path
        Args['result_path'] = result_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        Args['encoding_model_path'] = encoding_model_path
        Args['fmri_feature_corrs_path'] = fmri_feature_corrs_path
        Args['embedding_path'] = embedding_path
        Args['feature_path'] = feature_path
        Args['time_path'] = time_path
        Args['ref_time_path'] = ref_time_path
        Args['encoding_method'] = encoding_method
        Args['block_shuffle'] = block_shuffle
        Args['blocklen'] = blocklen
        Args['nfold'] = nfold
        Args['inner_fold'] = inner_fold
        Args['train_ratio'] = train_ratio
        Args['test_ratio'] = test_ratio
        Args['word_rate_method'] = word_rate_method
        Args['layer_num'] = layer_num
        Args['convolve_type'] = convolve_type
        Args['duration_time_type'] = duration_time_type
        Args['fmri_tr'] = fmri_tr
        Args['fmri_voxel_num'] = fmri_voxel_num
        Args['voxel_top_num'] = voxel_top_num
        Args['feature_type'] = feature_type
        Args['feature_abandon'] = feature_abandon
        Args['n_story'] = n_story
        Args['test_id'] = test_id
        Args['continuation_top_k'] = continuation_top_k
        Args['max_continuation'] = max_continuation
        Args['top_p'] = top_p
        Args['gpt_model_path'] = gpt_model_path
        Args['gpt_type'] = gpt_type
        Args['gpt_args'] = gpt_args
        Args['use_cuda'] = use_cuda
        Args['logging_file'] = logging_file
        Args['plot_brain'] = plot_brain
        Args['brain_template'] = brain_template

        logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M', )

        Args['ref_length'] = load_ref_TRs(ref_time_path)
        Args['story_range'] = range(1, n_story + 1)

        self.config = Args

if __name__ == '__main__':
    a = Album()
    print(a.config)