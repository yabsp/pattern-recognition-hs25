import torch
from audio_helper import stft, istft

@torch.no_grad()
def wiener_enhance_freqavg(noisy: torch.Tensor,
                           n_fft: int = 512,
                           hop: int = 160,
                           win: int = 320,
                           noise_quantile: float = 0.10,
                           gain_floor: float = 0.0,
                           eps: float = 1e-12) -> torch.Tensor:
    """
    Frequency-only Wiener gain G[u], estimated by aggregating across frames k.

    Inputs
    -------
    noisy : 1-D torch.Tensor [T] on any device/dtype (16 kHz expected by the rest of the code)
    n_fft, hop, win : STFT parameters
    noise_quantile : lower quantile over time used to estimate noise PSD per frequency (e.g., 0.10)
    gain_floor     : optional floor on the gain (e.g., 0.0 disables flooring)
    eps            : small positive number for numerical stability

    Returns
    -------
    enhanced : 1-D torch.Tensor [~T] with the same dtype/device as `noisy`
    """
    # ---- TODO: implement the frequency-averaged Wiener filter exactly as described in the sheet ----

    return 
