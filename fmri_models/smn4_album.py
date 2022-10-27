
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
        This class is used as the root_class of ALL classes in brain2char.
    '''
    def __init__(self, ):
        
        args = yaml.load(open(CONFIG), Loader=yaml.FullLoader)

        self.args = args

        self.fmri_path = args['path']['fmri_path']
        self.result_path = args['path']['result_path']
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.fmri_feature_corrs_path = args['path']['fmri_feature_corrs_path']
        self.embedding_path = args['path']['embedding_path']
        self.feature_path = args['path']['feature_path']
        self.time_path = args['path']['time_path']
        self.ref_time_path = args['path']['ref_time_path']
        self.encoding_method = args['encoding_model']['encoding_method']
        self.block_shuffle = args['encoding_model']['block_shuffle']
        self.blocklen = args['encoding_model']['blocklen']
        self.nfold = args['encoding_model']['nfold']
        self.inner_fold = args['encoding_model']['inner_fold']
        self.train_ratio = args['encoding_model']['train_ratio']
        self.test_ratio = args['encoding_model']['test_ratio']
        self.word_rate_type = args['word_rate']['word_rate_type']
        self.layer_num = args['feature_convolve']['layer_num']
        self.convolve_type = args['feature_convolve']['convolve_type']
        self.duration_time_type = args['feature_convolve']['duration_time_type']
        self.fmri_tr = args['data']['fmri_tr']
        self.fmri_tr_int = args['data']['fmri_tr_int']
        self.fmri_voxel_num = args['data']['fmri_voxel_num']
        self.voxel_top_num = args['data']['voxel_top_num']
        self.feature_type = args['data']['feature_type']
        self.feature_abandon = args['data']['feature_abandon']
        self.n_story = args['data']['n_story']
        self.test_id = args['data']['test_id']
        self.max_continuation = args['generate']['max_continuation']
        self.top_p = args['generate']['top_p']
        self.gpt_model_path = args['generate']['gpt_model_path']
        self.gpt_type = args['generate']['gpt_type']
        self.gpt_args = args['generate']['gpt_args']
        self.use_cuda = args['exp']['use_cuda']
        self.logging_file = args['report']['logging_file']
        self.plot_brain = args['report']['plot_brain']
        self.brain_template = args['report']['brain_template']

        logger = logging.getLogger()
        file_handler = logging.FileHandler(
            filename=self.logging_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M'))
        logger.addHandler(file_handler)
        self.log = logger

        self.ref_length = load_ref_TRs(args['path']['ref_time_path'])
        self.story_range = range(1, self.n_story + 1)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        pass