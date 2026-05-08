import numpy as np

def utterance_mean_std(feat: np.ndarray) -> np.ndarray:
    """
    feat: (T, D) -> emb: (2D,)
    """
    mu = feat.mean(axis=0)
    sd = feat.std(axis=0)
    return np.concatenate([mu, sd], axis=0)