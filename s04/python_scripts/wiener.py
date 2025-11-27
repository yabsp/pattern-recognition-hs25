import numpy as np
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
    # 1. Calculate STFT; 2. Collect Magnitudes across all frames -> new matrix with magnitudes
    # 3. S_ww = (quantile_k{|Y[k,u]|)², q = 10%; 4. Estimate S_yy 5. Estimate S_ww = max{S_yy - S_ww, 0}
    # 5. G[u] = S_xx[u] / (S_xx[u] + S_ww[u]) -> S[k,u] = G[u]Y[k,u]

    #print(f'Samples: {noisy.shape}')
    y_stft = stft(noisy, n_fft, hop, win)
    #print(f'Y: {y_stft.shape}')
    y_mag = y_stft.abs()
    #print(f'|Y|: {y_mag.shape}')
    s_ww_est = torch.quantile(y_mag, noise_quantile, dim=1).pow(2)
    #print(f'S_ww: {s_ww_est.shape}')
    s_yy_est = y_mag.pow(2).mean(dim=1)
    #print(f'S_yy: {s_yy_est.shape}')
    s_xx_est = torch.clamp(s_yy_est - s_ww_est, min=0.0)
    g_stft = s_xx_est / (s_xx_est + s_ww_est + eps)
    g_stft = torch.clamp(g_stft, min=gain_floor)
    #print(f'G: {g_stft.shape}')
    s = g_stft.unsqueeze(1) * y_stft

    enhanced = istft(s, n_fft, hop, win, noisy.shape[0])

    return enhanced
