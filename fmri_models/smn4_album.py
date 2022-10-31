
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
                all_args = args, 
                fmri_path = args['path']['fmri_path'], 
                result_path = args['path']['result_path'],
                encoding_model_path = args['path']['encoding_model_path'], 
                fmri_feature_corrs_path = args['path']['fmri_feature_corrs_path'],                
                embedding_path = args['path']['embedding_path'],
                feature_path = args['path']['feature_path'],
                time_path = args['path']['time_path'],
                ref_time_path = args['path']['ref_time_path'],

                encoding_method = args['encoding_model']['encoding_method'],
                block_shuffle = args['encoding_model']['block_shuffle'],
                blocklen = args['encoding_model']['blocklen'],
                nfold = args['encoding_model']['nfold'],
                inner_fold = args['encoding_model']['inner_fold'],
                train_ratio = args['encoding_model']['train_ratio'],
                test_ratio = args['encoding_model']['test_ratio'],

                word_rate_method = args['word_rate']['word_rate_method'],

                layer_num = args['feature_convolve']['layer_num'],
                convolve_type = args['feature_convolve']['convolve_type'],
                duration_time_type = args['feature_convolve']['duration_time_type'],
                fmri_tr = args['data']['fmri_tr'],
                fmri_tr_int = args['data']['fmri_tr_int'],
                fmri_voxel_num = args['data']['fmri_voxel_num'],
                voxel_top_num = args['data']['voxel_top_num'],
                
                feature_type = args['data']['feature_type'],
                feature_abandon = args['data']['feature_abandon'],
                n_story = args['data']['n_story'],
                test_id = args['data']['test_id'],
                
                continuation_top_k = args['generate']['continuation_top_k'], 
                max_continuation = args['generate']['max_continuation'],
                top_p = args['generate']['top_p'],
                gpt_model_path = args['generate']['gpt_model_path'],
                gpt_type = args['generate']['gpt_type'],
                gpt_args = args['generate']['gpt_args'],
                use_cuda = args['exp']['use_cuda'],
                logging_file = args['report']['logging_file'],
                plot_brain = args['report']['plot_brain'],
                brain_template = args['report']['brain_template'],
            **kwargs):

        self.args = all_args

        self.fmri_path = fmri_path
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.encoding_model_path = encoding_model_path
        self.fmri_feature_corrs_path = fmri_feature_corrs_path
        self.embedding_path = embedding_path
        self.feature_path = feature_path
        self.time_path = time_path
        self.ref_time_path = ref_time_path

        self.encoding_method = encoding_method
        self.block_shuffle = block_shuffle
        self.blocklen = blocklen
        self.nfold = nfold
        self.inner_fold = inner_fold
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.word_rate_method = word_rate_method

        self.layer_num = layer_num
        self.convolve_type = convolve_type
        self.duration_time_type = duration_time_type
        self.fmri_tr = fmri_tr
        self.fmri_tr_int = fmri_tr_int
        self.fmri_voxel_num = fmri_voxel_num
        self.voxel_top_num = voxel_top_num
        self.feature_type = feature_type
        self.feature_abandon = feature_abandon
        self.n_story = n_story
        self.test_id = test_id
        
        self.continuation_top_k = continuation_top_k
        self.max_continuation = max_continuation
        self.top_p = top_p
        self.gpt_model_path = gpt_model_path
        self.gpt_type = gpt_type
        self.gpt_args = gpt_args
        self.use_cuda = use_cuda
        self.logging_file = logging_file
        self.plot_brain = plot_brain
        self.brain_template = brain_template

        logger = logging.getLogger('simpleExample')
        logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M', )
        self.log = logger

        self.ref_length = load_ref_TRs(ref_time_path)
        self.story_range = range(1, n_story + 1)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        pass


if __name__ == '__main__':
    a = Album()
    print(a.log.info('test'))
    