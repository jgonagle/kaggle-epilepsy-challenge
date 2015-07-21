import math
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

def normalize(raw_signal):
    
    signal_float = np.array(raw_signal, np.float)
    
    signal_mean = np.mean(signal_float)
    signal_std = np.std(signal_float)
    norm_signal = (signal_float - signal_mean) / signal_std
    
    return [signal_mean, signal_std, norm_signal]

def bandpass(raw_signal, sample_rate, stop_low=.1, pass_low=.5, pass_high=50, stop_high=250, pass_gain=.95, stop_gain=.05):
    
    nyquist_freq = sample_rate / 2
    
    w_stop_low = stop_low / nyquist_freq
    w_pass_low = pass_low / nyquist_freq
    w_pass_high = pass_high / nyquist_freq
    w_stop_high = stop_high / nyquist_freq
    
    g_pass = -decibel(pass_gain)
    g_stop = -decibel(stop_gain)
    
    order, nat_freq = sps.buttord([w_pass_low, w_pass_high], [w_stop_low, w_stop_high], g_pass, g_stop)
    
    num_poly, denom_poly  = sps.butter(order, nat_freq, btype='bandpass')
    
    filtered_signal = sps.lfilter(num_poly, denom_poly, raw_signal)
    
    return filtered_signal

def decibel(ratio):
    
    return 10 * np.log(ratio, 10)
    
def spectrogram(raw_signal, sample_rate, frame_dur=1, jump_dur=.5, is_log=True):
    
    signal_mean, signal_std, norm_signal = normalize(raw_signal)
    
    signal_length = len(raw_signal)
    frame_length = math.ceil(sample_rate * frame_dur)
    frame_jump = math.floor(sample_rate * jump_dur)
    
    num_frames = (signal_length - frame_length) // frame_jump
    offset = signal_length - (frame_length + num_frames * frame_jump)
    
    stft_array = np.transpose(np.vstack([stft(norm_signal[i:i + frame_length]) for i in range(offset, num_frames * frame_jump, frame_jump)]))
    
    mag_stft = np.absolute(stft_array)
    phase_stft = np.angle(stft_array)
    
    if is_log:
        mag_stft = 2 * np.where(mag_cwt > 0, np.log(mag_cwt), 0)
    else:
        mag_stft = mag_stft**2
    
    return [mag_stft, phase_cwt, signal_mean, signal_std]

#uses morlet wavelet
def scaleogram(raw_signal, sample_rate, num_samps=128, log_low_freq=0, log_high_freq=1.6, num_freqs=32, width=6, sigma_factor=8, is_log=True):
    
    freqs = np.logspace(log_low_freq, log_high_freq, num_freqs, endpoint=True)
    sigma_array = width / (2. * np.pi * freqs)
    wavelet_len = min(math.ceil(np.max(sigma_array) * sample_rate * sigma_factor), len(raw_signal))
    scales = (freqs * wavelet_len) / (2. * width * sample_rate)
    
    wavelets = [sps.morlet(wavelet_len, width, s) for s in scales]
    
    time_indices = np.round(np.linspace(wavelet_len - 1, len(raw_signal) - 1, num_samps)).astype(np.dtype('int32'))
                        
    cwt_array = np.vstack([sample_cwt(raw_signal, w, time_indices) for w in wavelets])
    
    mag_cwt = np.absolute(cwt_array)
    phase_cwt = np.angle(cwt_array)
    
    if is_log:
        mag_cwt = 2 * np.where(mag_cwt > 0, np.log(mag_cwt), 0)
    else:
        mag_cwt = mag_cwt**2
    
    return [mag_cwt, phase_cwt]

def sample_cwt(raw_signal, wavelet, time_indices):
    
    return [point_convolve(raw_signal, wavelet, t) for t in time_indices]

#both functions start at index 0, n >= 0
def point_convolve(func_a, func_b, n):
    
    func_bp = func_b[0:n + 1]
    func_bp = func_bp[::-1]
    
    func_ap = func_a[max(n + 1 - len(func_bp), 0):min(len(func_a), n + 1)]
    func_bp = func_bp[0:len(func_ap)]

    return np.vdot(func_ap, func_bp)

#uses hanning window  
def stft(signal_frame):
    
    windowed_signal = signal_frame * np.hanning(len(signal_frame))
    stft_signal = np.fft.rfft(windowed_signal)
    
    return stft_signal

def visualize(s_gram, signal_name='1D Signal'):
    
    plt.xlabel('Sample Number')
    plt.ylabel('Frequency')
    plt.title(signal_name)
    plt.axis('equal')
    plt.pcolormesh(s_gram)
    
    plt.show()