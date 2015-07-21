import os
import scipy.io as sio
import numpy as np

import data_proc as dp
import load_save as ls

data_dir = '/home/jgonagle/Synced_Documents/Projects/Kaggle_Epilepsy_Challenge/data'
model_dir = '/home/jgonagle/Synced_Documents/Projects/Kaggle_Epilepsy_Challenge/models'

raw_data_dir = data_dir + '/raw'
scal_data_dir = data_dir + '/scaleograms'

subject_type_dict = {'Human':'Patient', 'Dog':'Dog'}
segment_type_set = {'interictal', 'preictal', 'test'}

pad_length = 4
raw_filetype = '.mat'
scal_filetype = '.csv'

def load_raw_data(subject_type, subject_num, segment_type, sample_num):
    
    subject_id = get_subject_id(subject_type, subject_num)
    subject_dir = get_raw_subject_dir(subject_type, subject_id)
    sample_name = get_sample_name(subject_id, segment_type, sample_num)
    
    data_filename = get_filename(subject_dir, sample_name, raw_filetype)
    
    if os.path.isfile(data_filename):
        mat_array = sio.loadmat(data_filename)
        raw_data = mat_array[''.join([segment_type, '_segment_', str(sample_num)])][0][0]
        
        return [raw_data, data_filename]
    
    return None

def load_scaleogram(freq, segment_type, train_index=None, subject_type=None, subject_num=None, sample_num=None, elec_index=None):
    
    scal_filename = get_scal_filename(freq, segment_type, train_index=train_index, subject_type=subject_type, subject_num=subject_num, sample_num=sample_num, elec_index=elec_index)
    
    if os.path.isfile(scal_filename):
        scaleogram = ls.load_csv(scal_filename)
        
        return [scaleogram, scal_filename]
    
    return None

def load_scaleogram(scal_filename):
    
    if os.path.isfile(scal_filename):
        scaleogram = ls.load_csv(scal_filename)
        
        return [scaleogram, scal_filename]
    
    return None

def save_scaleogram(scaleogram, freq, segment_type, train_index=None, subject_type=None, subject_num=None, sample_num=None, elec_index=None):

    scal_filename = get_scal_filename(freq, segment_type, train_index=train_index, subject_type=subject_type, subject_num=subject_num, sample_num=sample_num, elec_index=elec_index)
    
    ls.save_csv(scaleogram, scal_filename)

def load_all_test_data(freq):
    
    print(''.join(['\nBeginning load of all test data for frequency ', str(freq), ' Hz']))
    
    test_file_list = ls.get_file_list(get_scal_subject_dir(freq, 'test'))
    
    test_data = [load_scaleogram(t) for t in test_file_list]
    
    print(''.join(['Completed load of all test data for frequency ', str(freq), ' Hz']))
    
    return test_data

def load_all_training_data(freq):
    
    print(''.join(['\nBeginning load of all training data for frequency ', str(freq), ' Hz']))
    
    preictal_file_list = ls.get_file_list(get_scal_subject_dir(freq, 'preictal'))
    interictal_file_list = ls.get_file_list(get_scal_subject_dir(freq, 'interictal'))
    
    preictal_data = [load_scaleogram(p)[0] for p in preictal_file_list]
    interictal_data = [load_scaleogram(i)[0] for i in interictal_file_list]
    
    print(''.join(['Completed load of all training data for frequency ', str(freq), ' Hz']))
    
    return [preictal_data, interictal_data]

#returns random, balanced (i.e. equal numbers of preictal and interictal data per training set), scaled, paired (each scaleogram is paired with its label) training set
def yield_rbsp_train_data(preictal_data, interictal_data, num_samples, valid_frac=.05):
                
        train_data = preictal_data + preictal_data
        train_labels = [1 for i in range(len(preictal_data))] + [0 for i in range(len(interictal_data))]
        true_label_dist = [.5, .5]
        
        return dp.yield_rbsp_train_data(train_data, train_labels, true_label_dist, num_samples, valid_frac)

def save_theano_model(theano_model, train_params, freq):
    
    filename = ''.join([get_model_freq_dir(freq), '/CNN_Model_Validation_Error=', str(train_params[-1]), ])
    
    ls.save_timestamped_theano_model(theano_model, train_params, filename)

def load_theano_model(filename):
    
    return ls.load_theano_model(filename)

def load_latest_theano_model(freq):
    
    return ls.load_latest_theano_model(get_model_freq_dir(freq))
    
def assert_valid_sample_request(subject_type, subject_num, segment_type, sample_num):
    
    if subject_type not in subject_type_dict:
        raise Exception('Subject type needs to be in the set ' + str(set(subject_type_dict.keys())))
    
    if segment_type not in segment_type_set:
        raise Exception('Segment type needs to be in the set ' + str(segment_type_set))
    
    if subject_num <= 0:
        raise Exception('Subject number must be greater than zero')
    
    if sample_num <= 0:
        raise Exception('Sample number must be greater than zero')

def get_subject_id(subject_type, subject_num):
    
    return ''.join([subject_type_dict[subject_type], '_', str(subject_num)])

def get_raw_subject_dir(subject_type, subject_id):
    
    return ''.join([raw_data_dir, '/', subject_type, '/', subject_id])

def get_scal_subject_dir(freq, segment_type):
    
    return ''.join([scal_data_dir, '/', str(freq), '_hz/', segment_type])

def get_model_freq_dir(freq):
    
    return ''.join([model_dir, '/', str(freq), '_hz'])

def get_sample_name(subject_id, segment_type, sample_num):
    
    return ''.join([subject_id, '_', segment_type, '_segment_', zero_pad_left(pad_length, sample_num)])

def get_filename(directory, sample_name, filetype):
    
    return ''.join([directory, '/', sample_name, filetype])

def get_scal_filename(freq, segment_type, train_index=None, subject_type=None, subject_num=None, sample_num=None, elec_index=None):

    scal_dir = get_scal_subject_dir(freq, segment_type)
    
    if segment_type == 'test':
        subject_id = get_subject_id(subject_type, subject_num)
        sample_name = ''.join([get_sample_name(subject_id, segment_type, sample_num), '_', str(elec_index)])                
    else:
        sample_name = ''.join([segment_type, '_train_', str(train_index)])
    
    return get_filename(scal_dir, sample_name, scal_filetype)

def zero_pad_left(num_zeroes, val):

    return ''.join(['{:0>', str(num_zeroes), '}']).format(val)