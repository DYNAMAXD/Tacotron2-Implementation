import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

from tokenizer_hindi import HindiTokenizer
import config


import soundfile as sf
import torch



# ─────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────

def load_wav(path, sr=22050):
    audio, orig_sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = torch.tensor(audio, dtype=torch.float32)
    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)
    return audio.squeeze(0)


def amp_to_db(x, min_db=-100):
    clip_val = 10 ** (min_db / 20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val))


def db_to_amp(x):
    return 10 ** (x / 20)


def normalize(x, min_db=-100.0, max_abs_val=4.0):
    x = (x - min_db) / -min_db
    x = 2 * max_abs_val * x - max_abs_val
    return torch.clip(x, min=-max_abs_val, max=max_abs_val)


def denormalize(x, min_db=-100.0, max_abs_val=4.0):
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    return x * -min_db + min_db


# ─────────────────────────────────────────────
# Mel / Audio conversions
# ─────────────────────────────────────────────

class AudioMelConversions:
    def __init__(self,
                 num_mels=80,
                 sampling_rate=22050,
                 n_fft=1024,
                 window_size=1024,
                 hop_size=256,
                 fmin=0,
                 fmax=8000,
                 center=False,
                 min_db=-100.0,
                 max_scaled_abs=4.0):

        self.num_mels      = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft         = n_fft
        self.window_size   = window_size
        self.hop_size      = hop_size
        self.fmin          = fmin
        self.fmax          = fmax
        self.center        = center
        self.min_db        = min_db
        self.max_scaled_abs = max_scaled_abs

        mel_fb = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft,
                                     n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.spec2mel = torch.from_numpy(mel_fb).float()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def audio2mel(self, audio, do_norm=True):
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spec = torch.stft(
            input=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=torch.hann_window(self.window_size).to(audio.device),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.abs(spec)
        mel = torch.matmul(self.spec2mel.to(spec.device), spec)
        mel = amp_to_db(mel, self.min_db)
        if do_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)
        return mel

    def mel2audio(self, mel, do_denorm=True, griffin_lim_iters=60):
        if do_denorm:
            mel = denormalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)
        mel = db_to_amp(mel)
        spec = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()
        spec = np.maximum(0, spec)   # non-negative for Griffin-Lim
        audio = librosa.griffinlim(
            S=spec,
            n_iter=griffin_lim_iters,
            hop_length=self.hop_size,
            win_length=self.window_size,
            n_fft=self.n_fft,
            window="hann",
        )
        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        return audio.astype(np.int16)


# ─────────────────────────────────────────────
# Padding mask
# ─────────────────────────────────────────────

def build_padding_mask(lengths):
    B = lengths.size(0)
    T = int(torch.max(lengths).item())
    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1
    return mask.bool()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TTSDataset(Dataset):
    def __init__(self,
                 csv_path,
                 tokenizer,
                 sample_rate=22050,
                 n_fft=1024,
                 window_size=1024,
                 hop_size=256,
                 fmin=0,
                 fmax=8000,
                 num_mels=80,
                 min_db=-100.0,
                 max_scaled_abs=4.0):

        self.df         = pd.read_csv(csv_path)
        self.tokenizer  = tokenizer
        self.sample_rate = sample_rate

        self.audio_proc = AudioMelConversions(
            num_mels=num_mels,
            sampling_rate=sample_rate,
            n_fft=n_fft,
            window_size=window_size,
            hop_size=hop_size,
            fmin=fmin,
            fmax=fmax,
            min_db=min_db,
            max_scaled_abs=max_scaled_abs,
        )

        print(f"Loading dataset from {csv_path}  ({len(self.df)} samples)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = load_wav(row["file_path"], sr=self.sample_rate)
        mel   = self.audio_proc.audio2mel(audio, do_norm=True)    # (n_mels, T)
        return row["transcript"], mel.squeeze(0)


# ─────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────

def TTSCollator(tokenizer):
    def _collate_fn(batch):
        texts = [tokenizer.encode(b[0]) for b in batch]
        mels  = [b[1] for b in batch]

        input_lengths  = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels],  dtype=torch.long)

        # Sort by text length (longest first) — required for pack_padded_sequence
        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts  = [texts[i]  for i in sorted_idx]
        mels   = [mels[i]   for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        text_padded = torch.nn.utils.rnn.pad_sequence(
            texts, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        max_T    = int(output_lengths.max().item())
        num_mels = mels[0].shape[0]
        mel_padded  = torch.zeros(len(mels), num_mels, max_T)
        gate_padded = torch.zeros(len(mels), max_T)

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t - 1:] = 1.0

        mel_padded = mel_padded.transpose(1, 2)   # (B, T, n_mels)

        enc_mask = build_padding_mask(input_lengths)
        dec_mask = build_padding_mask(output_lengths)

        return text_padded, input_lengths, mel_padded, gate_padded, enc_mask, dec_mask

    return _collate_fn


# ─────────────────────────────────────────────
# Batch sampler (shuffles batch order, keeps similar lengths together)
# ─────────────────────────────────────────────

class BatchSampler:
    def __init__(self, dataset, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        n = len(dataset)
        indices = list(range(n))
        if drop_last:
            indices = indices[: (n // batch_size) * batch_size]
        batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
        perm = torch.randperm(len(batches)).tolist()
        self.batches = [batches[i] for i in perm]

    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)
