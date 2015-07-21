import os
import time
import pickle
import numpy as np

def save_pickled_file(pickle_me, filename):
    
    print('Beginning save of pickled version of', filename, '...')
    
    with open(filename, 'wb') as pickled_file:
        pickle.dump(pickle_me, pickled_file)
        
    print('Pickled save finished!')

def save_timestamped_pickled_file(pickle_me, filename):
    
    timestamped_filename = get_timestamped_filename(filename)
    save_pickled_file(pickle_me, timestamped_filename)

def load_existing_pickled_file(filename):
    
    been_pickled = None
    
    print('Beginning loading of saved pickled file', filename, '...')
    
    with open(filename, 'rb') as pickled_file:
        been_pickled = pickle.load(pickled_file)
        
    print('Pickled load finished!')
    
    return been_pickled

def load_latest_pickled_file(dir_name):
    
    file_list = get_file_list(dir_name)
    newest_filename = max(file_list, key = lambda a: os.stat(a).st_mtime)
    
    return load_existing_pickled_file(newest_filename)

def save_theano_model(theano_model, train_params, filename):
    
    model_train_params = [theano_model, train_params]
    
    timestamped_filename = get_timestamped_filename(filename)
    
    save_pickled_file(model_train_params, filename)

def save_timestamped_theano_model(theano_model, train_params, filename):
    
    timestamped_filename = get_timestamped_filename(filename)
    save_theano_model(theano_model, train_params, timestamped_filename)

def load_theano_model(filename):
    
    return load_existing_pickled_file(filename)

def load_latest_theano_model(dir_name):
    
    return load_latest_pickled_file(dir_name)

def save_csv(some_array, filename):
    
    np.savetxt(filename, some_array, delimiter=',')

def save_timestamped_csv(some_array, filename):
    
    timestamped_filename = get_timestamped_filename(filename)
    save_csv(some_array, timestamped_filename)

def load_csv(filename, skip_rows=0):
    
    return np.loadtxt(filename, delimiter=',', skiprows=skip_rows)

def get_file_list(dir_name):
    
    file_list = map(lambda a: ''.join([dir_name, '/', a]), os.listdir(os.path.abspath(dir_name)))
    #take out directories and hidden files
    file_list = filter(lambda a: not os.path.isdir(a) and os.path.split(a)[1][0] is not '.', file_list)
    
    return file_list

def get_timestamped_filename(filename):
    
    filename_pre, delimiter, filename_post = filename.rpartition('/')
    timestamped_filename = ''.join([filename_pre, delimiter, get_time_str(), '_', filename_post])
    
    return timestamped_filename

def get_time_str():
    
    return '_'.join(time.ctime(time.time()).split(' '))