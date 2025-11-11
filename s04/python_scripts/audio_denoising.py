from pathlib import Path
import random, re
import numpy as np
import torch, torchaudio
from torch.utils.data import Dataset, DataLoader

from audio_helper import TARGET_SR, to_mono, ensure_sr, pad_or_crop, snr_db, save_wav
from wiener import wiener_enhance_freqavg
# Optional CNN import for comparison:
# from models import TinyDenoiser

# ------------ Dataset (VoiceBank + DEMAND) ------------

class VoiceBankDemandDataset(Dataset):
    def __init__(self, root: Path, split: str, segment_sec: float = None, sr: int = TARGET_SR):
        assert split in {"train", "test"}
        if split == "train":
            cdir = root/"clean_trainset_28spk_wav"
            ndir = root/"noisy_trainset_28spk_wav"
        else:
            cdir = root/"clean_testset_wav"
            ndir = root/"noisy_testset_wav"

        self.sr = sr
        self.segment_len = int(segment_sec*sr) if segment_sec else None

        clean_paths = sorted(list(cdir.rglob("*.wav")) + list(cdir.rglob("*.WAV")))
        noisy_paths = sorted(list(ndir.rglob("*.wav")) + list(ndir.rglob("*.WAV")))

        # Robust matching by filename, then stem, then relaxed DEMAND-suffix stripping
        noisy_by_name = {p.name: p for p in noisy_paths}
        pairs = [(cp, noisy_by_name[cp.name]) for cp in clean_paths if cp.name in noisy_by_name]

        if not pairs:
            noisy_by_stem = {p.stem: p for p in noisy_paths}
            pairs = [(cp, noisy_by_stem[cp.stem]) for cp in clean_paths if cp.stem in noisy_by_stem]

        if not pairs:
            def strip_suffix(stem: str):
                return re.sub(r"_(babble|cafe|street|white|pink|engine|airport)(_[\-+]?\d+dB)?$", "", stem)
            bucket = {}
            for p in noisy_paths:
                bucket.setdefault(strip_suffix(p.stem), []).append(p)
            pairs = [(cp, bucket[strip_suffix(cp.stem)][0]) for cp in clean_paths
                     if strip_suffix(cp.stem) in bucket and len(bucket[strip_suffix(cp.stem)]) == 1]

        if not pairs:
            raise RuntimeError("Could not match clean/noisy files. Check dataset layout.")

        self.pairs = pairs
        self.split = split

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        cpath, npath = self.pairs[i]
        c, sr_c = torchaudio.load(cpath)
        n, sr_n = torchaudio.load(npath)
        c = ensure_sr(to_mono(c), sr_c, self.sr)
        n = ensure_sr(to_mono(n), sr_n, self.sr)
        assert c.numel() == n.numel(), f"Length mismatch: {npath.name}"

        if self.segment_len and self.split == "train":
            if c.numel() > self.segment_len:
                start = random.randint(0, c.numel()-self.segment_len)
                c = c[start:start+self.segment_len]
                n = n[start:start+self.segment_len]
            else:
                c = pad_or_crop(c, self.segment_len)
                n = pad_or_crop(n, self.segment_len)

        # Normalize to ~ -26 dBFS for comfortable listening
        rms = torch.sqrt(torch.mean(n**2) + 1e-12)
        target_rms = 10**(-26/20)
        if rms > 0:
            scale = target_rms / rms
            n = n*scale; c = c*scale
        return n, c

# ------------ Evaluation helpers ------------

@torch.no_grad()
def eval_wiener_and_export(test_loader, n_examples=5, noise_quantile=0.10, out_dir=Path("out")):
    out_dir.mkdir(parents=True, exist_ok=True)
    dsnrs = []
    done = 0
    for noisy, clean in test_loader:
        noisy, clean = noisy.squeeze(0), clean.squeeze(0)
        est = wiener_enhance_freqavg(noisy, noise_quantile=noise_quantile)
        L = min(clean.numel(), noisy.numel(), est.numel())
        c, y, xw = clean[:L], noisy[:L], est[:L]
        dsnrs.append(float(snr_db(c, xw) - snr_db(c, y)))
        if done < n_examples:
            base = out_dir/f"wiener_ex{done:02d}"
            save_wav(base.with_name(base.name+"_noisy.wav"),  y)
            save_wav(base.with_name(base.name+"_wiener.wav"), xw)
            save_wav(base.with_name(base.name+"_clean.wav"),  c)
            done += 1
    print(f"ΔSNR (Wiener, q={noise_quantile:.2f}): {np.mean(dsnrs):.2f} dB | Saved {done} demo sets to {out_dir}")

def main():
    torch.manual_seed(1337); np.random.seed(1337)
    DATA_ROOT = Path("")  # <- set to your VoiceBank+DEMAND root
    test_ds  = VoiceBankDemandDataset(DATA_ROOT, "test", segment_sec=None, sr=TARGET_SR)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)

    # Calls the student's function:
    eval_wiener_and_export(test_loader, n_examples=6, noise_quantile=0.10, out_dir=Path("out"))

if __name__ == "__main__":
    main()
