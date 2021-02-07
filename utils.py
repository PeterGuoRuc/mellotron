import numpy as np
from scipy.io.wavfile import read
import torch
import librosa
import soundfile as sf

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    if torch.cuda.is_available():
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    else:
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path, required_sampling_rate):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = sf.read(full_path, dtype="float32")
    if sampling_rate != required_sampling_rate:
        data = librosa.resample(data, sampling_rate, required_sampling_rate)
        sampling_rate = required_sampling_rate
    data = (data * 32768).astype("int")
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate
        


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
