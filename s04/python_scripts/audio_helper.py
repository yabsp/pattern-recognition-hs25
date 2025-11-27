from pathlib import Path
import torch, torchaudio
import numpy as np

TARGET_SR = 16000

def to_mono(x):  # [C,T] -> [T]
    return x.mean(dim=0) if x.dim()==2 else x

def ensure_sr(wav, sr, target=TARGET_SR):
    if sr == target: return wav
    return torchaudio.functional.resample(wav, sr, target)

def pad_or_crop(x, length):
    if x.numel() == length: return x
    if x.numel() > length:  # center-crop
        start = (x.numel() - length)//2
        return x[start:start+length]
    out = torch.zeros(length, dtype=x.dtype, device=x.device)
    out[:x.numel()] = x
    return out

def snr_db(ref, est, eps=1e-10):
    num = torch.sum(ref**2) + eps
    den = torch.sum((ref - est)**2) + eps
    return 10.0*torch.log10(num/den)

def save_wav(path: Path, wav: torch.Tensor, sr: int = TARGET_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().unsqueeze(0).clamp(-1, 1)
    torchaudio.save(str(path), wav, sample_rate=sr)

def stft(x: torch.Tensor, n_fft=512, hop=160, win=320) -> torch.Tensor:
    w = torch.hann_window(win, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=w, return_complex=True)

def istft(X: torch.Tensor, n_fft=512, hop=160, win=320, length=None) -> torch.Tensor:
    w = torch.hann_window(win, device=X.device)
    return torch.istft(X, n_fft=n_fft, hop_length=hop, win_length=win,
                       window=w, length=length)
