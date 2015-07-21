import math
import random
import numpy as np

def find_scale_transform(data):
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    return [mean, std]

#mean and std are over axis=0
def apply_scale(some_data, mean, std):
    
    m_std_data = np.array(some_data)
    n_mean = np.array(mean)
    n_std = np.array(std)
    #want to make sure not to divide by zero if the standard deviation is zero
    n_std += (n_std == 0)
        
    m_std_data -= n_mean
    m_std_data /= n_std
    
    return m_std_data

def scale_data(some_data):
    
    mean, std = find_scale_transform(some_data)
    
    m_std_data = apply_scale(some_data, mean, std)
    
    return m_std_data

#rbsp stands for random (i.e. the rbsp set contains random picks from each training data class), balanced (i.e. training data sample's label reflects the true label distribution), scaled (for scaled data), and paired (so that each sample is paired with its label).  every call to this function yields a new rbsp set of training data as well as a rbsp set of validation data. train_classes is a list of lists of training examples for each class
def yield_rbsp_train_data(train_data, train_labels, true_label_dist, num_samples, valid_frac=.05):
    
    scaled_train_data = scale_data(train_data)
    paired_data =list(zip(scaled_train_data, train_labels))
    
    sample_classes = [list(filter(lambda l: l[1]==i, paired_data)) for i in range(len(true_label_dist))]
    #necessary to ensure the validation and training classes are randomly selected from the total training data
    for sc in sample_classes: np.random.shuffle(sc)
    
    size_valid_class_pool = [math.ceil(len(sc) * valid_frac) for sc in sample_classes]
    
    valid_sample_classes = [sc[0:svp] for sc, svp in zip(sample_classes, size_valid_class_pool)]
    train_sample_classes = [sc[svp:] for sc, svp in zip(sample_classes, size_valid_class_pool)]
    
    num_valid_samples = math.ceil(num_samples * valid_frac)
    num_train_samples = num_samples - num_valid_samples
    
    num_valid_class_samples = [math.ceil(num_valid_samples * lp) for lp in true_label_dist]
    num_train_class_samples = [math.ceil(num_train_samples * lp) for lp in true_label_dist]
    
    #since the number of samples required from each class may be larger than the class size, efficient sampling is from the "multiplied" list (i.e. self-concatenation), replicating sampling with replacement
    valid_class_multipliers = [math.ceil(nv / len(vc)) for nv, vc in zip(num_valid_class_samples, valid_sample_classes)]
    train_class_multipliers = [math.ceil(nt / len(tc)) for nt, tc in zip(num_train_class_samples, train_sample_classes)]
    
    while True:
        sampled_valid_classes = [random.sample(vc * vm, nv) for vc, vm, nv in zip(valid_sample_classes, valid_class_multipliers, num_valid_class_samples)]
        sampled_train_classes = [random.sample(tc * tm, nt) for tc, tm, nt in zip(train_sample_classes, train_class_multipliers, num_train_class_samples)]
        
        sampled_valid_rbsp = [vs for svc in sampled_valid_classes for vs in svc]
        sampled_train_rbsp = [ts for stc in sampled_train_classes for ts in stc]
        
        valid_rbsp = random.sample(sampled_valid_rbsp, num_valid_samples)
        train_rbsp = random.sample(sampled_train_rbsp, num_train_samples)
        
        yield [valid_rbsp, train_rbsp]
