# Inference
import os
import matplotlib
import matplotlib.pyplot as plt

import sys
################################################################################################
sys.path.append('.')             
sys.path.append('waveglow/')   
################################################################################################

from itertools import cycle
import numpy as np
import scipy as sp
from scipy.io.wavfile import write
import soundfile as sf
import pandas as pd
import librosa
import torch

from hparams import create_hparams
from model import Tacotron2
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
################################################################################################
from train_utils import load_model 
################################################################################################
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
from mellotron_utils import get_data_from_musicxml

hparams = create_hparams()

    

stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)

def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]

def plot_mel_f0_alignment(mel_source, mel_outputs_postnet, f0s, alignments, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto', origin='bottom', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()

def load_mel(path):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    if torch.cuda.is_available():
        melspec = melspec.cuda()
    return melspec


def agumentation(arpabet_dict, audio_paths, target_spk_id_list, output_path, ljs = False):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Step1: Basic Setups

    if not ljs:
        # Whether to use lj speech
        checkpoint_path = "mellotron_libritts.pt"
    else:
        checkpoit_path = "mellotron_ljs.pt"
    if torch.cuda.is_available():
        tacotron = load_model(hparams).cuda().eval()
    else:
        tacotron = load_model(hparams).eval()
    tacotron.load_state_dict(torch.load(checkpoint_path, map_location="cpu")['state_dict'])


    waveglow_path = 'waveglow_256channels_v4.pt'
    if torch.cuda.is_available():
        waveglow = torch.load(waveglow_path)['model'].cuda().eval()
        denoiser = Denoiser(waveglow).cuda().eval()
    else:
        waveglow = torch.load(waveglow_path, map_location="cpu")['model'].eval().cpu()
        denoiser = Denoiser(waveglow).eval()


    arpabet_dict = cmudict.CMUDict(arpabet_dict)
    dataloader = TextMelLoader(audio_paths, hparams)
    datacollate = TextMelCollate(1)

    # Step2: Load 
    for file_idx in range(len(dataloader)):
        source_scp = open(os.path.join(output_path, "source.scp"), "w", encoding="utf-8")

        audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]
        source_scp.write("{} {}\n".format(file_idx, audio_path))

        # get audio path, encoded text, pitch contour and mel for gst
        text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :]
        pitch_contour = dataloader[file_idx][3][None]
        if torch.cuda.is_available():
            text_encoded = text_encoded.cuda()
            pitch_contour = pitch_contour.cuda()
        mel = load_mel(audio_path)
        # load source data to obtain rhythm using tacotron 2 as a forced aligner
        x, y = tacotron.parse_batch(datacollate([dataloader[file_idx]]))

        # Step3: Perform speaker transfer
        with torch.no_grad():
            # get rhythm (alignment map) using tacotron 2
            mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = tacotron.forward(x)
            rhythm = rhythm.permute(1, 0, 2)

        for spk_id in target_spk_id_list:
            speaker_id = torch.LongTensor([spk_id])

            if torch.cuda.is_available():
                speaker_id = speaker_id.cuda()

            with torch.no_grad():
                mel_outputs, mel_outputs_postnet, gate_outputs, _ = tacotron.inference_noattention(
                    (text_encoded, mel, speaker_id, pitch_contour * 0.4, rhythm))

            with torch.no_grad():
                audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]



            sf.write(os.path.join(output_path, "{}-{}.wav".format(file_idx, spk_id)), audio.detach().cpu().numpy().T, hparams.sampling_rate)


def test():
    agumentation("data/cmu_dictionary", 'data/examples_filelist.txt', [0, 1, 2], "augmentation")


if __name__ == "__main__":
    test()