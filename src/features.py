from typing import Literal
import numpy as np
import librosa


FeatureType = Literal["mfcc", "mfcc_d_dd"]


def extract_features(
    file_path: str,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    do_trim: bool = True,
    trim_top_db: int = 25,
    do_cmvn: bool = True,
    feature_type: FeatureType = "mfcc",
) -> np.ndarray:
    """
    返回 (T, D)：
    - mfcc: D=n_mfcc
    - mfcc_d_dd: D=3*n_mfcc (MFCC + Δ + ΔΔ)
    """
    y, _ = librosa.load(file_path, sr=sr)

    # 找出能量高于阈值的一段连续区间保留，去掉前后静音
    if do_trim:
        y, _ = librosa.effects.trim(y, top_db=trim_top_db)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )  # (n_mfcc, T)

    if feature_type == "mfcc":
        feat = mfcc
    elif feature_type == "mfcc_d_dd":
        d1 = librosa.feature.delta(mfcc, order=1) # 特征的一阶时间变化率（类似速度）
        d2 = librosa.feature.delta(mfcc, order=2) # 特征的二阶时间变化率（类似加速度）
        feat = np.vstack([mfcc, d1, d2])  # (3*n_mfcc, T) , 维度从 20 变 60
    else:
        raise ValueError(f"Unknown feature_type={feature_type}")

    # 每条语音的特征维度可能不同，训练 GMM 前需要做 CMVN（均值方差归一化）减少录音增益/通道差异
    if do_cmvn:
        feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-8)

    return feat.T  # (T, D)