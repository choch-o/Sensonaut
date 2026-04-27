# utils/audio_features.py
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
from typing import Tuple, Optional

# ----------------- Small DSP helpers -----------------

def _frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n = x.shape[0]
    if n < frame_len:
        return np.zeros((0, frame_len), dtype=x.dtype)
    num = 1 + (n - frame_len) // hop_len
    idx = np.arange(0, frame_len)[None, :] + np.arange(0, num * hop_len, hop_len)[:, None]
    return x[idx]

def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(x**2) + eps))

def _butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = max(1.0, low) / nyq
    high = min(high, nyq - 1) / nyq
    b, a = sig.butter(order, [low, high], btype="band")
    return b, a

def _gcc_phat(sig_r: np.ndarray, sig_l: np.ndarray, fs: int, max_tau: float) -> Tuple[float, np.ndarray]:
    """
    GCC-PHAT on (R frame, L frame). Returns (tau_seconds, cc),
    where tau is R→L delay (sec). Negative tau => Right leads.
    """
    n = sig_r.shape[0] + sig_l.shape[0]
    N = 1 << (n - 1).bit_length()
    Xr = np.fft.rfft(sig_r, n=N)
    Xl = np.fft.rfft(sig_l, n=N)
    R = Xr * np.conj(Xl)
    R /= np.abs(R) + 1e-15
    cc = np.fft.irfft(R, n=N)

    max_shift = int(min(int(N / 2), np.floor(max_tau * fs)))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    tau = shift / float(fs)  # seconds
    return tau, cc

def _robust_stats(x: np.ndarray):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": np.nan, "mad": np.nan}
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return {"median": med, "mad": mad}

# ----------------- Public API -----------------

def compute_itd_ild_from_wav(
    wav_path: str,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    max_itd_ms: float = 1.2,
    vad_db: Optional[float] = None,
    bands: Tuple[Tuple[float, float], ...] = ((20.0, 20000.0),),  # first band is "wide"
    preemph: bool = False,
) -> Tuple[float, float]:
    """
    Compute (ITD_seconds, ILD_dB) from a stereo WAV using frame-wise GCC-PHAT and ILD.
    - ITD sign: R→L delay (sec). Negative -> Right leads.
    - ILD sign: dB. Positive -> Right louder.

    Args:
        wav_path: path to stereo WAV.
        frame_ms, hop_ms: STFT-like analysis framing (Hann window).
        max_itd_ms: |tau| search bound for GCC (covers human head range).
        vad_db: if set (e.g., -40), exclude low-energy frames by simple RMS dBFS VAD.
        bands: tuple of (low_Hz, high_Hz). The FIRST band is used for ILD summary.
        preemph: simple pre-emphasis (0.97) if True.

    Returns:
        (itd_seconds, ild_db)
    """
    fs, data = wav.read(wav_path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("WAV must be stereo (2 channels).")
    # Normalize integer PCM -> [-1, 1]
    if data.dtype.kind in "iu":
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)

    L = data[:, 0].copy()
    R = data[:, 1].copy()
    if preemph:
        L = sig.lfilter([1, -0.97], [1], L)
        R = sig.lfilter([1, -0.97], [1], R)

    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop_len   = int(round(hop_ms   * 1e-3 * fs))
    win = sig.get_window("hann", frame_len, fftbins=True)

    L_frames = _frame_signal(L, frame_len, hop_len) * win
    R_frames = _frame_signal(R, frame_len, hop_len) * win
    num_frames = L_frames.shape[0]
    if num_frames == 0:
        return 0.0, 0.0

    # Simple VAD mask (optional)
    vad_mask = np.ones(num_frames, dtype=bool)
    if vad_db is not None:
        for i in range(num_frames):
            e = 0.5 * (_rms(L_frames[i]) ** 2 + _rms(R_frames[i]) ** 2)
            db = 10.0 * np.log10(e + 1e-12)
            vad_mask[i] = (db >= vad_db)

    # ---- ITD via GCC-PHAT (per frame) ----
    max_tau = max_itd_ms * 1e-3
    itd_sec = np.zeros(num_frames, dtype=float)
    gcc_peak = np.zeros(num_frames, dtype=float)
    for i in range(num_frames):
        tau, cc = _gcc_phat(R_frames[i], L_frames[i], fs, max_tau)  # R->L
        itd_sec[i] = tau
        gcc_peak[i] = float(np.max(np.abs(cc)))

    # keep only valid (VAD) frames
    keep = vad_mask & np.isfinite(itd_sec)
    if not np.any(keep):
        keep = np.isfinite(itd_sec)
    itd_keep = itd_sec[keep]
    peak_keep = gcc_peak[keep] + 1e-12

    # Weighted median by GCC peak (more stable than plain median)
    if itd_keep.size:
        sorter = np.argsort(itd_keep)
        x = itd_keep[sorter]
        w = peak_keep[sorter] / np.sum(peak_keep)
        cdf = np.cumsum(w)
        itd_sec_summary = float(x[np.searchsorted(cdf, 0.5)])
    else:
        itd_sec_summary = 0.0

    # ---- ILD (use first band as "wide") ----
    lo, hi = bands[0]
    b, a = _butter_bandpass(lo, hi, fs, order=4)
    Lb = sig.lfilter(b, a, L)
    Rb = sig.lfilter(b, a, R)
    Lbf = _frame_signal(Lb, frame_len, hop_len) * win
    Rbf = _frame_signal(Rb, frame_len, hop_len) * win

    ild = np.zeros(num_frames, dtype=float)
    for i in range(num_frames):
        # +dB => Right louder
        ild[i] = 20.0 * np.log10((_rms(Rbf[i]) + 1e-12) / (_rms(Lbf[i]) + 1e-12))

    ild_summary = float(np.median(ild[keep])) if np.any(keep) else float(np.median(ild))

    return itd_sec_summary, ild_summary