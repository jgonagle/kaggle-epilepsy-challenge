import math

import epi_dp_ls as edl
import sig_proc as sp

scal_height = 32
scal_width = 128

log_low_freq = 0
log_high_freq = 1.6
width = 6
sigma_factor = 8
is_log = True

def gen_scal_subject(subject_type, subject_num, preictal_start=1, interictal_start=1):
    
    subject_id = edp.get_subject_id(subject_type, subject_num)
    
    print(''.join(['\tBeginning generation of all scaleograms for ', subject_id, '\'s data']))
    
    for segment_type in edp.segment_type_set:
        print(''.join(['\t\tBeginning generation of scaleograms for ', subject_id, '\'s ', segment_type, ' data']))
        
        train_index = None
        
        if segment_type == 'preictal':
            train_index = preictal_start
        elif segment_type == 'interictal':
            train_index = interictal_start
        
        sample_num = 1
        valid_sample_num = True
        
        while valid_sample_num:
            raw_data = edp.load_raw_data(subject_type, subject_num, segment_type, sample_num)
            
            if raw_data is None:
                valid_sample_num = False
            else:
                raw_data = raw_data[0]
                
                sample_rate = math.ceil(raw_data[2][0][0])
                num_electrodes = len(raw_data[0])
                
                for elec_index in range(0, num_electrodes):
                    print(''.join(['\t\t\tGenerating scaleogram for ', subject_id, '\'s ', segment_type, ' segment ', str(sample_num), ', electrode ', str(elec_index + 1), ' signal']))
                    scaleogram = sp.scaleogram(raw_data[0][elec_index], sample_rate, num_samps=scal_width, log_low_freq=log_low_freq, log_high_freq=log_high_freq, num_freqs=scal_height, width=width, sigma_factor=sigma_factor, is_log=is_log)[0]
                    
                    #once all processing done, change elec_index=elec_index to elec_index=elec_index+1
                    edp.save_scaleogram(scaleogram, sample_rate, segment_type, train_index=train_index, subject_type=subject_type, subject_num=subject_num, sample_num=sample_num, elec_index=elec_index)
                    
                    if segment_type == 'preictal' or segment_type == 'interictal':
                        train_index += 1
                
            sample_num += 1
        
        print(''.join(['\t\tFinished generating all scaleograms for ', subject_id, '\'s ', segment_type, ' data']))
    
    print(''.join(['\tFinished generating all scaleograms for ', subject_id, '\'s data'])) 