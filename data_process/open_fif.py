import mne
import matplotlib.pyplot as plt

# 该fif文件存放地址

fname = 'derivatives_preprocessed_data_sub-01_MEG_sub-01_task-RDR_run-1_meg.fif'

raw = mne.io.read_raw_fif(fname)
print(raw)
print(raw.info)