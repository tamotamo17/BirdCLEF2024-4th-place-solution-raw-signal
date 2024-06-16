import random
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from nnAudio.Spectrogram import CQT1992v2, CQT2010v2
import torch

def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(
        y=y, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec

def compute_qtransform(y):
    qtransform = CQT1992v2(sr=32000, fmin=256, n_bins=160, hop_length=250, output_format='Magnitude',
                           norm=1, window='tukey',bins_per_octave=27, verbose=False)
    img = qtransform(torch.Tensor(y))
    img = (64*img + 1).log().numpy()[0]
    return img

def crop_or_pad(y, length, sr, train=True, probs=None):
    """
    Crops an array to a chosen length
    Arguments:
        y {1D np array} -- Array to crop
        length {int} -- Length of the crop
        sr {int} -- Sampling rate
    Keyword Arguments:
        train {bool} -- Whether we are at train time. If so, crop randomly, else return the beginning of y (default: {True})
        probs {None or numpy array} -- Probabilities to use to chose where to crop (default: {None})
    Returns:
        1D np array -- Cropped array
    """
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (
                    np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            )
            start = int(sr * (start))

        y = y[start: start + length]

    return y.astype(np.float32)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

class WaveformDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 df_sep: pd.DataFrame,
                 target_columns: list,
                 transforms: dict,
                 duration: int=5,
                 secondary_ratio: float=0.0,
                 use_first: bool=True,
                 downsample: int=2,
                 mode='train'):
        self.df = df
        self.df_sep = df_sep
        self.target_columns = target_columns
        self.duration = duration
        self.secondary_ration = secondary_ratio
        self.use_first = use_first
        self.downsample = downsample
        self.mode = mode

        if mode == 'train':
            self.wave_transforms = transforms['train'] 
        else:
            self.wave_transforms = transforms['valid']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        samples_sep = self.df_sep.loc[self.df_sep['filename']==sample['filename'], :]
        if self.use_first or (self.mode!='train'):
            try:
                sample_sep = samples_sep.loc[samples_sep['clip_file_path'].str.contains('_0.ogg')].iloc[0]
            except:
                sample_sep = samples_sep.iloc[0]
        else:
            sample_sep = samples_sep.iloc[np.random.randint(len(samples_sep))]

        wav_path = sample_sep["clip_file_path"]
        labels = sample["new_target"]
        rating = sample["rating_0_1"]

        y = sf.read(wav_path)[0]
        if y.ndim!=1:
            y = y.mean(1)

        if len(y) > 0:
            y = y[:self.duration*SR]

            if self.wave_transforms:
                y = self.wave_transforms(y, sr=SR)

        y = np.concatenate([y, y, y])[:self.duration * SR]
        y = crop_or_pad(y, self.duration * SR, sr=SR, train=self.mode, probs=None)
        if self.mode=='train':
            s = random.randint(0, self.downsample-1)
        else:
            s = 0
        y = y[s:len(y):self.downsample]
        
        targets = np.zeros(len(self.target_columns), dtype=float)
        labels_list = labels.split()
        for i, ebird_code in enumerate(labels_list):
            if i==0:
                targets[self.target_columns.index(ebird_code)] = 1.0
            else:
                try:
                    targets[self.target_columns.index(ebird_code)] = self.secondary_ratio
                except:
                    pass
        return {
            "path": wav_path,
            "image": y,
            "targets": targets,
            "rating": rating
        }