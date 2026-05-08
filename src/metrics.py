from __future__ import annotations
from typing import List, Tuple
from collections import Counter

import numpy as np


def per_speaker_recall(y_true: List[str], y_pred: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
    - speakers: (S,)
    - recall: (S,)  每个 speaker 的召回率 = TP/(TP+FN)
    """
    speakers = np.array(sorted(set(y_true)))
    true_cnt = Counter(y_true)
    tp_cnt = Counter([t for t, p in zip(y_true, y_pred) if t == p])

    recall = np.array([tp_cnt[s] / true_cnt[s] for s in speakers], dtype=float)
    return speakers, recall


def worst_k_speakers(y_true: List[str], y_pred: List[str], k: int = 20):
    speakers, recall = per_speaker_recall(y_true, y_pred)
    idx = np.argsort(recall)[:k]
    return speakers[idx], recall[idx]