from __future__ import annotations
from collections import Counter
from typing import List, Tuple

def top_confusions(y_true: List[str], y_pred: List[str], top_k: int = 20) -> List[Tuple[str, str, int]]:
    """
    返回最常见的混淆对 (true, pred, count)，排除预测正确的样本。
    """
    c = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            c[(t, p)] += 1
    return [(t, p, n) for (t, p), n in c.most_common(top_k)]