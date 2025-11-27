# +
import numpy as np
from pathlib import Path
import torch
import random, math, os
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from audio_helper import stft, istft, snr_db, save_wav


# A simple convnet denoiser operating in the log-magnitude STFT domain
OUT_DIR   = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SR = 16000
SEGMENT_SEC = 2.0       
RIDGE = 1e-3                   
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0   
PERSISTENT = False
seed = 1337
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

# A simple convnet denoiser operating in the log-magnitude STFT domain

class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, 1, F, T]
        z = self.enc(x)
        m = self.dec(z)
        return m

def stft_mag_phase(x, n_fft=512, hop=160, win=320, eps=1e-8):
    X = stft(x, n_fft, hop, win)
    mag = torch.sqrt(X.real**2 + X.imag**2 + eps)
    phase = torch.atan2(X.imag, X.real)
    return mag, phase

def magphase_to_complex(mag, phase):
    return torch.polar(mag, phase)

def train_epoch(model, loader, opt, scaler=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.train()
    model.to(device)
    total = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        M_mag, M_phase = stft_mag_phase(noisy)     # [B,F,T]
        C_mag, _       = stft_mag_phase(clean)     # [B,F,T] (same length)

        logM = torch.log1p(M_mag).unsqueeze(1)     # [B,1,F,T]

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=("cuda" in device)):
            mask = model(logM)                      # [B,1,F',T']
            if mask.shape[-2:] != logM.shape[-2:]:
                mask = F.interpolate(mask, size=logM.shape[-2:], mode='bilinear', align_corners=False)
            mask = mask.squeeze(1)                  # [B,F,T]

            est_mag = torch.expm1(logM.squeeze(1)) * mask
            # loss = F.l1_loss(est_mag, C_mag)
            loss = F.l1_loss(torch.log1p(est_mag), torch.log1p(C_mag))

        opt.zero_grad(set_to_none=True)
        if scaler is not None and "cuda" in device:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

        total += float(loss.detach())
    return total / max(len(loader), 1)

@torch.no_grad()
def enhance_cnn_full(model, mixture, device='cpu'):
    model.eval().to(device)
    x = mixture.to(device)
    M_mag, M_phase = stft_mag_phase(x)                   # [F,T]
    logM = torch.log1p(M_mag).unsqueeze(0).unsqueeze(0)  # [1,1,F,T]
    mask = model(logM)                                   # [1,1,F',T']
    if mask.shape[-2:] != logM.shape[-2:]:
        mask = F.interpolate(mask, size=logM.shape[-2:], mode='bilinear', align_corners=False)
    mask = mask.squeeze()                                  # [F,T]
    est_mag = torch.expm1(logM.squeeze()) * mask
    S = magphase_to_complex(est_mag, M_phase)
    out = istft(S)
    return out.cpu()

def get_accel_device(prefer: str = "auto") -> str:
    """
    Returns 'cuda', 'mps', or 'cpu'.
    prefer can be 'cuda' or 'mps' to force a choice if available; otherwise choose best available.
    """
    has_cuda = torch.cuda.is_available()
    has_mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if prefer == "cuda" and has_cuda:
        return "cuda"
    if prefer == "mps" and has_mps:
        return "mps"
    if prefer != "auto":
        # fall back if the preferred isn't available
        pass

    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"

def describe_device(device: str) -> str:
    if device == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
        return f"Using CUDA on {name}"
    if device == "mps":
        return "Using Apple Silicon MPS (Metal)"
    return "Using CPU"

def train_cnn_denoiser(train_loader, epochs=EPOCHS, lr=LR, device: str = None,
                       save_path=OUT_DIR/"tinydenoiser_vb.pt"):
    if device is None:
        device = get_accel_device("auto")
    print(describe_device(device))


    model = TinyDenoiser().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on {len(train_loader.dataset)} pairs for {epochs} epochs. Device={device}")
    for epoch in range(1, epochs+1):
        loss = train_epoch(model, train_loader, opt, scaler=None, device=device)
        print(f"[Epoch {epoch}/{epochs}] loss={loss:.4f}")

    torch.save({"model": model.state_dict()}, save_path)
    print(f"Saved checkpoint to: {save_path}")
    return model, device

@torch.no_grad()
def eval_cnn_and_export(model, test_loader, n_examples=5, out_dir=OUT_DIR, device: str = None):
    if device is None:
        device = get_accel_device("auto")
    model.eval().to(device)

    dsnr_cnn, examples_done = [], 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for idx, (noisy, clean) in enumerate(test_loader):
        noisy, clean = noisy.squeeze(0), clean.squeeze(0)
        est = enhance_cnn_full(model, noisy, device=device)
        L = min(clean.numel(), est.numel(), noisy.numel())
        c = clean[:L]; y = noisy[:L]; xc = est[:L]

        dsnr_c = float(snr_db(c, xc) - snr_db(c, y))
        dsnr_cnn.append(dsnr_c)

        if examples_done < n_examples:
            base = f"cnn_ex{examples_done:02d}"
            save_wav(out_dir/f"{base}_noisy.wav", y)
            save_wav(out_dir/f"{base}_cnn.wav",   xc)
            save_wav(out_dir/f"{base}_clean.wav", c)
            examples_done += 1

    print(f"ΔSNR (CNN): {np.mean(dsnr_cnn):.2f} dB")
    print(f"Saved {examples_done} demo sets to: {out_dir}")
    return float(np.mean(dsnr_cnn))
# -


